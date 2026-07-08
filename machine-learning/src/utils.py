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