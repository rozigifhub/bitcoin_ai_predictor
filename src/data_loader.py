import pandas as pd

def load_data(path):
    """
    Load dataset Bitcoin dari CSV
    """
    df = pd.read_csv(path)

    return df
