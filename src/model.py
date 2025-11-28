import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(window_size=30, n_features=5):
    """
    Membangun model LSTM untuk prediksi harga Bitcoin.
    - window_size: jumlah hari sebelumnya
    - n_features: jumlah fitur (open, high, low, price, volume)
    """

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1)  # output: harga (scaled)
    ])

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["mae"]
    )

    return model
