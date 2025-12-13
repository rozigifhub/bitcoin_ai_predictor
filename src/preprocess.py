import pandas as pd

def clean_data(df):
    #hapus quote,hapus spasi,dan lowercase
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace('"', '')
    #ganti nama
    df = df.rename(columns ={
        "price" : "close",
        "vol." : "volume"
    })
    # waktu jadikan format date
    df["date"] = pd.to_datetime(df["date"])
    # list kolom numerik yang mau dibersihkan
    list_column = ["close", "open", "high", "low", "volume"]
    #hapus format
    for col in list_column:
        if df[col].dtype == "object":
                df[col] = (
                    df[col].str.replace(",", "", regex = False)
                    .str.replace("K", "e3",regex = False)
                    .str.replace("M", "e6",regex = False)
                    .str.replace("B", "e9").astype(float)
                )

    #ascending time
    df = df.sort_values("date", ascending = True).reset_index(drop = True)
    return df


def select_column(df):
    #kolom yang dipake
    return df[["date", "open", "high", "low", "close", "volume"]]

def normalize_data(df, *, fit_end=None, scaler=None, clip=False):
    df = df.copy()

    kolom_numerik = ["open", "high", "low", "close", "volume"]

    if fit_end is not None and not (0 < fit_end <= len(df)):
        raise ValueError(f"fit_end harus antara 1..len(df), dapat: {fit_end}")

    # 1) FIT: hitung min/max dari subset train (baris awal)
    if scaler is None:
        scaler = {}
        df_fit = df.iloc[:fit_end] if fit_end is not None else df

        for kolom in kolom_numerik:
            nilai_min = df_fit[kolom].min()
            nilai_max = df_fit[kolom].max()
            scaler[kolom] = {"min": float(nilai_min), "max": float(nilai_max)}

    # 2) TRANSFORM: pakai scaler (train) untuk normalisasi df ini
    for kolom in kolom_numerik:
        nilai_min = scaler[kolom]["min"]
        nilai_max = scaler[kolom]["max"]
        range_data = nilai_max - nilai_min

        if range_data == 0:
            df[kolom] = 0.0
        else:
            df[kolom] = (df[kolom] - nilai_min) / range_data
            if clip:
                df[kolom] = df[kolom].clip(0.0, 1.0)

    return df, scaler

