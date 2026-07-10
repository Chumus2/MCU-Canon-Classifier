import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from utils import load_data, normalize_genres

from pathlib import Path
import warnings


warnings.filterwarnings("ignore")


def FE(data_path: str, output_path: str):
    df = load_data(data_path)

    # ratings from different sources already exist in normalized form (IMDb, TMDB, etc.),
    # so raw rating columns become redundant and would duplicate the same signal
    similar = {
        "imdb_rating": "IMDb",
        "rt_score": "Rotten Tomatoes",
        "metacritic_score": "Metacritic",
        "tmdb_rating": "TMDB",
    }
    df = df.drop(columns=list(similar.keys()))

    # title/top_actor are identifiers or high-cardinality text features with no generalizable signal
    df = df.drop(columns=["title", "top_actor"])

    # categorical columns will be one-hot encoded; language/country dropped due to high cardinality
    categorial_cols = ["type", "rated", "mcu_phase", "universe", "decade", "status", "language", "country"]

    # language/country create too many sparse dummy columns = drop instead of encoding
    df = df.drop(columns=["language", "country"])
    categorial_cols.remove("language")
    categorial_cols.remove("country")

    # one-hot encoding for categorical features; drop_first avoids perfect multicollinearity
    df = pd.get_dummies(df, columns=categorial_cols, drop_first=True)

    # high-cardinality entity columns (names of people/keywords) would explode feature space
    # and cause overfitting - safer to drop without embedding/text processing
    unique_cols = ["director", "writer", "actors", "directors", "producers", "composer", "top5_cast", "tmdb_keywords"]
    df = df.drop(columns=unique_cols)

    # free-text fields require NLP; since this is a tabular model, drop them for now
    text_cols = ["plot", "awards", "tagline", "document"]
    df = df.drop(columns=text_cols)

    # normalize genre naming across sources (TMDB vs others) to avoid duplicate categories
    genre_synonyms = {
        "Science Fiction": "Sci-Fi",
        "Sci-Fi & Fantasy": "Sci-Fi, Fantasy",
        "Action & Adventure": "Action, Adventure",
        "Talk": "Talk-Show",
        "Kids": "Family",
    }
    df["tmdb_genres_norm"] = df["tmdb_genres"].apply(lambda x: normalize_genres(x, genre_synonyms))
    df["genre_norm"] = df["genre"].apply(lambda x: normalize_genres(x, {}))

    # combine genres from multiple sources into a unified set per movie
    df["all_genres"] = df.apply(lambda row: list(set(row["genre_norm"]) | set(row["tmdb_genres_norm"])), axis=1)
    # "TV Movie" flag is a strong format indicator not captured well by genres alone
    df["is_tv_movie_format"] = df["tmdb_genres"].str.contains("TV Movie").astype(int)
    # drop intermediate genre columns after merging
    df = df.drop(columns=["tmdb_genres", "genre", "tmdb_genres_norm", "genre_norm"])

    # convert multi-label genre list into binary features
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(df["all_genres"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df.drop(columns=["all_genres"]), genre_encoded], axis=1)

    # these columns are complex nested structures already partially encoded elsewhere
    df = df.drop(columns=["production_countries", "spoken_languages"])

    # universe and mcu_phase directly encode the target (MCU membership) = data leakage
    # keeping them would let the model "cheat" instead of learning real patterns
    leaky_cols = [col for col in df.columns if col.startswith(("universe_", "mcu_phase_"))]
    df = df.drop(columns=leaky_cols)

    # low-correlation or noisy features identified during EDA = removed to reduce overfitting
    USELESS_COLUMNS = [
        "cast_count", "tmdb_votes", "imdb_votes",
        "genre_Action", "genre_Adventure", "genre_Animation",
        "genre_Drama", "genre_Fantasy", "genre_Musical",
        "genre_War", "genre_Family", "genre_Unknown"
    ]
    df = df.drop(columns=USELESS_COLUMNS)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    try:
        df = FE(
            data_path="../data/cleaned-data/cleaned-data.csv",
            output_path="../data/processed-data/processed-data.csv"
        )

        print("==========DF CHECK==========")
        print(df.shape)
        print(df.head(5))
        print(df.info())
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise