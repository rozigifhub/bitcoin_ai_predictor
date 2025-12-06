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

def normalize_data(df):
     
    df = df.copy()

    scaler = {}

    kolom_numerik = ["open", "high", "low", "close", "volume"]

    for kolom in kolom_numerik:
        nilai = df[kolom]
        nilai_max = nilai.max()
        nilai_min = nilai.min()

        # simpan scaler
        scaler[kolom] = {"min": nilai_min, "max": nilai_max}

        range_data = nilai_max - nilai_min
        if range_data == 0:
            df[kolom] = 0.0
        else:
            #normalisasi
            df[kolom] = (nilai - nilai_min) / range_data

    return df, scaler
