import pandas as pd


def eta_squared(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
) -> tuple[float, int]:
    """Hitung eta-squared antara peubah kategorikal dan numerik.

    Eta-squared mengukur proporsi ragam peubah respon yang dijelaskan oleh
    group_col — analog dengan R^2 dari one-way ANOVA. Nilai 0 berarti
    group_col tidak menjelaskan ragam peubah respon sama sekali, nilai 1
    berarti peubah respon sepenuhnya ditentukan oleh grup.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        group_col (str): Nama kolom kategorikal sebagai peubah pengelompok.
        target_col (str): Nama kolom numerik sebagai peubah respon.

    Returns:
        tuple[float, int]: Sebuah tuple yang berisi:
            - float: Nilai eta-squared dalam rentang [0, 1].
            - int: Jumlah kategori unik di group_col (setelah dropna).
    """
    sub = df[[group_col, target_col]].dropna()
    grand_mean = sub[target_col].mean()

    ss_total = ((sub[target_col] - grand_mean) ** 2).sum()
    ss_between = (
        sub.groupby(group_col)[target_col]
        .apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2)
        .sum()
    )

    return ss_between / ss_total, sub[group_col].nunique()
