import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data, normalize_row

from pathlib import Path
import warnings


warnings.filterwarnings("ignore")

ACCENT_PALETTE = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#79c0ff', '#f85149', '#56d4dd']

pd.set_option({
    "display.max_rows": None,
    "display.max_columns": None,
    "display.width": None,
    "display.max_colwidth": None,
})

plt.rcParams.update({
    'figure.facecolor': '#0d1117',

    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'axes.titlecolor': '#e6edf3',
    'axes.prop_cycle': plt.cycler(color=ACCENT_PALETTE),

    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',

    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#cccccc',
    'grid.linestyle': "--",
    'grid.alpha': 0.75,

    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',

    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 11,
})

sns.set_palette(ACCENT_PALETTE)


def EDA(cast_path: str, master_path: str, rag_path: str, ratings_path: str, output_path: str):
    cast = load_data(cast_path)
    master = load_data(master_path)
    rag = load_data(rag_path)
    ratings = load_data(ratings_path)

    # profile_path is just an image URL with no analytical value = drop instead of imputing
    cast = cast.drop(columns=["profile_path"])
    # missing character name usually means "uncredited/extra role", not a data error
    cast["character"] = cast["character"].fillna("Unknown")

    # raw "score" mixes 3 different formats depending on source (e.g. "84%", "7.4/10", "7.4"),
    # so normalize_row puts everything on a single 0-10 scale before sources can be compared/merged
    ratings["score_normalized"] = ratings.apply(normalize_row, axis=1)
    ratings = ratings.drop(columns=["score"])
    # fill missing scores per-source, not globally: IMDb/RT/Metacritic have different
    # score distributions, so a global mean would bias sources with fewer ratings
    ratings["score_normalized"] = ratings.groupby("source")["score_normalized"].transform(
        lambda x: x.fillna(x.mean())
    )

    # external-service IDs and link/media fields = no predictive/analytical value here
    USELESS_COLUMNS = [
        "tmdb_id", "imdb_id", "poster_url", "website", "production"
    ]
    master = master.drop(columns=USELESS_COLUMNS)
    # mostly "N/A"/single-film "collections" (checked via value_counts) =  not useful
    master = master.drop(columns=["collection_name"])

    # season_count/episode_count are NaN specifically for movies (not TV) -> 0 is the
    # correct value here, not a "missing" one
    master["season_count"] = master["season_count"].fillna(0)
    master["episode_count"] = master["episode_count"].fillna(0)
    master = master.drop(columns=["network"], errors="ignore")

    # these look numeric but are stored as strings with formatting ("84%", "72/100",
    # "1,144") = strip formatting first, THEN convert, or to_numeric would fail
    master["rt_score"] = master["rt_score"].astype(str).str.replace("%", "", regex=False)
    master["rt_score"] = pd.to_numeric(master["rt_score"], errors="coerce")

    master["metacritic_score"] = master["metacritic_score"].astype(str).str.split("/").str[0]
    master["metacritic_score"] = pd.to_numeric(master["metacritic_score"], errors="coerce")

    master["imdb_votes"] = master["imdb_votes"].astype(str).str.replace(",", "", regex=False)
    master["imdb_votes"] = pd.to_numeric(master["imdb_votes"], errors="coerce")

    # computed AFTER the string-numeric conversion above, so rt_score/metacritic_score/
    # imdb_votes get picked up as numeric_cols and filled with median (not "Unknown")
    numeric_cols = master.select_dtypes(include="number").columns.tolist()
    categorical_cols = master.select_dtypes(include="object").columns.tolist()

    for col in numeric_cols:
        master[col] = master[col].fillna(master[col].median())

    for col in categorical_cols:
        master[col] = master[col].fillna("Unknown")

    # tmdb_id/actor_tmdb_id are just external identifiers, not analytical features
    cast = cast.drop(columns=['tmdb_id', 'actor_tmdb_id'], errors='ignore')
    # rag's own row id isn't a reliable join key across all 4 tables (title+year is
    # the only key shared by every dataset) -> safe to drop
    rag = rag.drop(columns=["id"], errors='ignore')

    # a real score of exactly 0.0 is implausible for TMDB specifically -> treat as a
    # missing-value placeholder and re-impute with the median (more robust to outliers
    # than the mean used in the first pass above)
    ratings.loc[(ratings["source"] == "TMDB") & (ratings["score_normalized"] == 0.0), "score_normalized"] = np.nan
    ratings["score_normalized"] = ratings.groupby("source")["score_normalized"].transform(
        lambda x: x.fillna(x.median())
    )

    # runtimes under 10 min / over 200 min are almost certainly data entry errors
    # for feature films -> treat as missing and impute rather than keep as real values
    master.loc[master["runtime_min"] < 10, "runtime_min"] = np.nan
    master.loc[master["runtime_min"] > 200, "runtime_min"] = np.nan
    master["runtime_min"] = master["runtime_min"].fillna(master["runtime_min"].median())

    # a real budget/revenue of exactly $0 doesn't happen for released films = these
    # are "unknown", not "free", so treat as missing before imputing
    master.loc[master["revenue_usd"] == 0, "revenue_usd"] = np.nan
    master.loc[master["budget_usd"] == 0, "budget_usd"] = np.nan
    master["revenue_usd"] = master["revenue_usd"].fillna(master["revenue_usd"].median())
    master["budget_usd"] = master["budget_usd"].fillna(master["budget_usd"].median())

    # money/vote columns are heavily right-skewed (a handful of blockbusters dominate),
    # so log1p compresses the scale for modeling/plots; log1p (not log) is safe near zero
    master["revenue_log"] = np.log1p(master["revenue_usd"])
    master["budget_log"] = np.log1p(master["budget_usd"])
    master["box_office_log"] = np.log1p(master["box_office_usd"])

    master["imdb_votes"] = np.log1p(master["imdb_votes"])
    master["tmdb_votes"] = np.log1p(master["tmdb_votes"])
    # drop raw (non-log) versions now that *_log columns replace them, to avoid feeding
    # both the skewed and transformed version downstream
    master = master.drop(columns=['revenue_usd', 'budget_usd', 'box_office_usd'], errors='ignore')

    # title+year is the only key shared by all 4 tables (cast/ratings have no id at all),
    # so every merge below joins on that pair rather than any internal id column
    df = master.merge(rag[["title", "year", "document"]], on=["title", "year"], how="left")

    # cast has multiple rows per film (one per actor) = must aggregate to one row
    # per (title, year) first, or the merge below would duplicate every master row
    cast_agg = (
        cast.groupby(["title", "year"])
        .agg(cast_count=("actor_name", "count"),
             top_actor=("actor_name", "first"))
        .reset_index()
    )

    # same one-to-many issue for ratings (one row per source) = pivot sources into
    # their own columns so each film again becomes a single row
    rating_wide = ratings.pivot_table(
        index=["title", "year"], columns="source", values="score_normalized"
    ).reset_index()

    df = df.merge(cast_agg, on=["title", "year"], how="left")
    df = df.merge(rating_wide, on=["title", "year"], how="left")

    # NaNs here come from left-joins that found no match (e.g. no rating recorded
    # for that source, or no cast entry) = impute per-column so it doesn't cascade
    for col in ["IMDb", "TMDB", "Metacritic", "Rotten Tomatoes"]:
        df[col] = df[col].fillna(df[col].median())

    # cast_count=0 means "no cast data found", not "zero actors" = keep in mind if
    # this feeds a model later, since it's a placeholder rather than a true zero
    df["cast_count"] = df["cast_count"].fillna(0)
    df["top_actor"] = df["top_actor"].fillna("Unknown")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == '__main__':
    try:
        df = EDA(
            cast_path="../data/raw-data/raw-marvel-cast.csv",
            master_path="../data/raw-data/raw-marvel-master.csv",
            rag_path="../data/raw-data/raw-marvel-rag.csv",
            ratings_path="../data/raw-data/raw-marvel-ratings.csv",
            output_path="../data/cleaned-data/cleaned-data.csv"
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise