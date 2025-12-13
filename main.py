from src.data_loader import load_data
from src.preprocess import clean_data, select_column, normalize_data
from src.window import create_window
import numpy as np



def main():
    df_raw = load_data("data/bitcoin.csv")

    df_clean = clean_data(df_raw)
    df_select = select_column(df_clean)
    split_raw = int(len(df_select) * 0.8)
    df_normal, scaler = normalize_data(df_select, fit_end=split_raw)

    print(df_normal.head())
    print("Normalized shape:", df_normal.shape)
    print("Scaler:", scaler)

    #ubah jadi numpy array
    df_matrix = df_normal[["open","high","low","close","volume"]].values
    X_train, y_train, X_test, y_test = create_window(df_matrix, split_index=split_raw) 


    np.set_printoptions(precision=4, suppress=True)

    print("X_train: ", X_train.shape)
    print("X_train visual: ")
    print(X_train[1])
    print("1 window shape:", X_train[0].shape)
    print("1 timestep shape:", X_train[0, 0].shape)
    #print("y_train: ", y_train.shape)
    #print("x_test: ", X_test.shape)
    #print("y_test: ", y_test.shape)
    



if __name__ == "__main__":
    main()