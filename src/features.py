import pandas as pd

DEFAULT_MISSING_COLS = [
    "property_acidity_index",
    "latitude",
    "longitude",
    "cation_Ca",
    "cation_Mg",
    "cation_exchange_capacity",
    "cation_Na",
]


def add_missing_indicators(
    df: pd.DataFrame,
    cols: list[str] = DEFAULT_MISSING_COLS,
    suffix: str = "_is_missing",
) -> pd.DataFrame:
    """Tambahkan kolom indikator biner untuk setiap kolom di `cols`.
    Mengembalikan DataFrame baru, tidak mengubah input asli.
    """
    df = df.copy()
    for col in cols:
        df[f"{col}{suffix}"] = df[col].isna().astype(int)
    return df
