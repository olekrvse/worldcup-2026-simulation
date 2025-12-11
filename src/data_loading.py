# src/data_loading.py

import pandas as pd
from .config import DATA_RAW_DIR

def load_results():
    path = f"{DATA_RAW_DIR}/results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def load_former_names():
    path = f"{DATA_RAW_DIR}/former_names.csv"
    try:
        df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    except FileNotFoundError:
        df = None
    return df
