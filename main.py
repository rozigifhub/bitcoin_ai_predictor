from src.data_loader import load_data
from src.preprocess import clean_data, select_column, normalize_data
from src.window import create_window



def main():
    df_raw = load_data("data/bitcoin.csv")

    df_clean = clean_data(df_raw)
    df_select = select_column(df_clean)
    df_normal, scaler = normalize_data(df_select)

    print(df_normal.head())
    print("Normalized shape:", df_normal.shape)
    print("Scaler:", scaler)

    #ubah jadi numpy array
    df_matrix = df_normal[["open","high","low","close","volume"]].values
    X_train, y_train, X_test, y_test = create_window(df_matrix) 

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", X_test.shape)
    print("y_test: ", y_test.shape)
    



if __name__ == "__main__":
    main()