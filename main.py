from src.data_loader import load_data
from src.preprocess import clean_data, select_columns, normalize_data
from src.window import create_window

def main():
    df_raw = load_data("data/bitcoin.csv")

    df_clean = clean_data(df_raw)
    df_selected = select_columns(df_clean)

    print(df_raw.columns)

    data_scaled, scaler = normalize_data(df_selected)

    X_train, y_train, X_test, y_test = create_window(data_scaled)

    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape X_test:", X_test.shape)
    print("Shape y_test:", y_test.shape)


if __name__ == "__main__":
    main()
