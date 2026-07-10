import numpy as np
import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} does not exist')

    return pd.read_csv(data_path)


def normalize_row(row):
    value = row["score"]
    source = row["source"]

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if "%" in value:
        return float(value.replace("%", "")) / 10
    elif "/" in value:
        num, denom = value.split("/")
        return float(num) / float(denom) * 10
    else:
        return float(value)


def normalize_genres(genre_str, synonyms):
    parts = [g.strip() for g in genre_str.split(",")]
    normalized = []
    for p in parts:
        replacement = synonyms.get(p, p)
        normalized.extend(r.strip() for r in replacement.split(","))
    return list(set(normalized))