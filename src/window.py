import numpy as np

def create_window(data_scaled, window_size=30):
    """
    Membuat window time-series untuk LSTM.
    Input: data_scaled shape (n, 5)
    - open, high, low, price, volume

    Output:
    - X (samples, window_size, features)
    - y (samples,)
    """

    X = []
    y = []

    
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i])  # 30 hari sebelumnya
        y.append(data_scaled[i, 3])  # kolom ke-3 = price (Close)

    X = np.array(X)
    y = np.array(y)

    # split train test 80%
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test