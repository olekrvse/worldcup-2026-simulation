# src/fixtures_wc2026.py

import os
import pandas as pd

from .preprocessing import normalize_team_names

# Base directory = project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(BASE_DIR, "data", "fixtures")


def load_group_stage_fixtures(filename: str = "wc2026_group_stage.csv") -> pd.DataFrame:
    """
    Load the World Cup 2026 group stage fixtures and normalize team names
    so they match the model's canonical naming (e.g. IR Iran -> Iran).
    """
    path = os.path.join(FIXTURES_DIR, filename)
    df = pd.read_csv(path, parse_dates=["date"])

    # Normalize team names to match training data (Iran, South Korea, Ivory Coast, Cape Verde, etc.)
    df = normalize_team_names(df)

    return df
