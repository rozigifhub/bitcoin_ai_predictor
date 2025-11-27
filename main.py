from src.data_loader import load_data
from src.preprocess import clean_data, select_columns

def main():
    df_raw = load_data("data/bitcoin.csv")

    df_clean = clean_data(df_raw)
    df_selected = select_columns(df_clean)

    print("5 data teratas(setelah cleaning):")
    print(df_clean.head())

    print("\nInfo dataset:")
    print(df_selected.info())

if __name__ == "__main__":
    main()
