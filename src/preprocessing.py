import pandas as pd

CATEGORY_MAPPINGS = {
    "has_band_A_spectrum": ["YES", "NO"],
    "has_band_B_spectrum": ["YES", "NO"],
    "sampling_strategy": ["Auger", "Profile", "Unknown"],
}


def set_categorical_dtypes(df):
    df = df.copy()

    for col, categories in CATEGORY_MAPPINGS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)

    return df
