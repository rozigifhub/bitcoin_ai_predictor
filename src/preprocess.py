import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """
    Bersihkan kolom-kolom utama Bitcoin:
    - hilangkan kutip
    - hilangkan koma
    - convert ke float
    """

    # bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()

    # rename kolom agar konsisten
    df = df.rename(columns={
        "Date": "date",
        "Price": "price",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume"
    })

    # convert date
    df["date"] = pd.to_datetime(df["date"])

    # list kolom numerik yang mau dibersihkan
    numeric_cols = ["price", "open", "high", "low", "volume"]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('K', '000', regex=False)
        )

    # convert jadi float
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # urutkan berdasarkan tanggal
    df = df.sort_values("date").reset_index(drop=True)

    return df


def select_columns(df):
    """
    Pilih kolom yang akan dipakai model LSTM.
    """
    return df[["date", "open", "high", "low", "price", "volume"]]

def normalize_data(df):
    """
    Normalisasi fitur numerik (open, high, low, price, volume)
    Output:
    - data_scaled: numpy array 2D
    - scaler: objek scaler untuk inverse transform nanti
    """

    scaler = MinMaxScaler()

    # ambil hanya kolom numerik
    values = df[["open", "high", "low", "price", "volume"]].values

    # scaling (fit + transform)
    data_scaled = scaler.fit_transform(values)

    return data_scaled, scaler