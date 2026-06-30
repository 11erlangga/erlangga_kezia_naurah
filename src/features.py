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
    """Tambahkan kolom indikator biner untuk missing value pada daftar kolom tertentu.

    Fungsi ini akan membuat kolom baru yang bernilai 1 jika data pada kolom asli
    kosong (NaN), dan 0 jika ada datanya. Mengembalikan DataFrame baru tanpa
    mengubah input asli.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data awal.
        cols (list[str], optional): Daftar nama kolom yang akan dicek missing value-nya.
            Defaults to DEFAULT_MISSING_COLS.
        suffix (str, optional): Akhiran teks yang akan ditambahkan pada nama kolom baru.
            Defaults to "_is_missing".

    Returns:
        pd.DataFrame: DataFrame baru yang sudah ditambahkan kolom-kolom indikator.
    """
    df = df.copy()
    for col in cols:
        df[f"{col}{suffix}"] = df[col].isna().astype(int)
    return df


def add_band_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur jumlah band spektral yang tersedia.

    Menghitung total ketersediaan band berdasarkan kolom `has_band_A_spectrum`
    dan `has_band_B_spectrum`. Nilai yang dihasilkan berupa: 0 (tidak ada band),
    1 (satu band), atau 2 (kedua band tersedia). Mengembalikan DataFrame baru tanpa
    mengubah input asli.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data awal. Harus memiliki kolom
            `has_band_A_spectrum` dan `has_band_B_spectrum`.

    Returns:
        pd.DataFrame: DataFrame baru dengan tambahan kolom `n_bands_available`.
    """
    df = df.copy()
    df["n_bands_available"] = df["has_band_A_spectrum"] + df["has_band_B_spectrum"]
    return df


def add_cation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur rasio kation Ca terhadap Mg.

    Rumus yang digunakan adalah `cation_Ca` / `cation_Mg`. Baris dengan nilai
    `cation_Mg` = 0 atau NaN akan menghasilkan NaN dan dibiarkan seperti itu
    karena akan ditangani oleh pipeline di tahap berikutnya. Mengembalikan
    DataFrame baru tanpa mengubah input asli.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data awal. Harus memiliki kolom
            `cation_Ca` dan `cation_Mg`.

    Returns:
        pd.DataFrame: DataFrame baru dengan tambahan kolom `cation_Ca_to_Mg_ratio`.
    """
    df = df.copy()
    df["cation_Ca_to_Mg_ratio"] = df["cation_Ca"] / df["cation_Mg"]
    return df
