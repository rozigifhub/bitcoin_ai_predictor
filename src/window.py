import numpy as np

def create_window(data_normal, window_size=30, split_index=None):
    X_train, y_train, X_test, y_test = [], [], [], []

    if split_index is None:
        split_index = int(len(data_normal) * 0.8)

    if split_index <= window_size:
        raise ValueError("split_index harus > window_size")

    for i in range(window_size, split_index):
        X_train.append(data_normal[i - window_size:i])
        y_train.append(data_normal[i, 3])  # close

    for i in range(split_index, len(data_normal)):
        X_test.append(data_normal[i - window_size:i])
        y_test.append(data_normal[i, 3])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
