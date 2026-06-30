from src.features import add_band_availability, add_cation_ratio, add_missing_indicators


def prepare_inference_data(df):
    """Menerapkan pipeline feature engineering yang sama dengan training.

    Urutan transformasi penting. Missing indicators harus dibuat sebelum
    cation ratio agar indikator missingness untuk ``cation_Ca`` dan
    ``cation_Mg`` merefleksikan nilai asli, bukan nilai yang menjadi
    ``NaN`` akibat operasi pembagian.

    Args:
        df (pd.DataFrame):
            Raw DataFrame (misalnya hasil memuat ``test.csv``). DataFrame
            boleh masih mengandung kolom non-fitur seperti ``sample_id``.
            Fungsi ini tidak membuang kolom apa pun, hanya menambahkan
            fitur baru.

    Returns:
        pd.DataFrame:
            DataFrame dengan fitur tambahan. Input asli tidak diubah.
    """
    df = add_missing_indicators(df)
    df = add_band_availability(df)
    df = add_cation_ratio(df)
    return df
