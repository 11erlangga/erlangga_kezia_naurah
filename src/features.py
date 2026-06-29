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


def add_band_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur jumlah band spektral yang tersedia.

    n_bands_available = has_band_A_spectrum + has_band_B_spectrum
    Nilai: 0 (tidak ada band), 1 (satu band), 2 (kedua band tersedia).

    Mengembalikan DataFrame baru, tidak mengubah input asli.
    """
    df = df.copy()
    df["n_bands_available"] = df["has_band_A_spectrum"] + df["has_band_B_spectrum"]
    return df


def add_cation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur rasio kation Ca terhadap Mg.

    cation_Ca_to_Mg_ratio = cation_Ca / cation_Mg
    Baris dengan cation_Mg = 0 atau NaN akan menghasilkan NaN (dibiarkan,
    karena pipeline sudah menangani missing value di tahap berikutnya).

    Mengembalikan DataFrame baru, tidak mengubah input asli.
    """
    df = df.copy()
    df["cation_Ca_to_Mg_ratio"] = df["cation_Ca"] / df["cation_Mg"]
    return df
