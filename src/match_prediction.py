# src/match_prediction.py

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import DATA_PROCESSED_DIR


@lru_cache(maxsize=1)
def load_team_params() -> pd.DataFrame:
    """
    Load team-level attack/defence parameters from the Poisson model.
    Cached so it's only read from disk once.
    """
    path = os.path.join(DATA_PROCESSED_DIR, "team_params_poisson.csv")
    df = pd.read_csv(path)
    return df


def _get_global_params(df: pd.DataFrame) -> tuple[float, float]:
    """
    Get the intercept and home advantage (same for all teams).
    """
    intercept = float(df["intercept"].iloc[0])
    home_advantage = float(df["home_advantage"].iloc[0])
    return intercept, home_advantage


def _get_team_row(df: pd.DataFrame, team: str) -> pd.Series:
    """
    Return the row in df corresponding to the given team.
    """
    row = df.loc[df["team"] == team]
    if row.empty:
        raise ValueError(f"Team '{team}' not found in team_params_poisson.csv")
    return row.iloc[0]


def expected_goals(home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
    """
    Compute expected goals for home and away team using the fitted Poisson model.

    Poisson log-rate structure:
        log(λ_home) = intercept + attack_home + defence_away + (home_advantage if not neutral)
        log(λ_away) = intercept + attack_away + defence_home
    """
    df = load_team_params()
    intercept, home_advantage = _get_global_params(df)

    home_row = _get_team_row(df, home_team)
    away_row = _get_team_row(df, away_team)

    attack_home = float(home_row["attack"])
    defence_home = float(home_row["defence"])
    attack_away = float(away_row["attack"])
    defence_away = float(away_row["defence"])

    # log-lambdas
    log_lambda_home = intercept + attack_home + defence_away
    log_lambda_away = intercept + attack_away + defence_home

    # apply home advantage if not neutral
    if not neutral:
        log_lambda_home += home_advantage

    lambda_home = float(np.exp(log_lambda_home))
    lambda_away = float(np.exp(log_lambda_away))

    return lambda_home, lambda_away


def match_outcome_probabilities(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    max_goals: int = 10,
) -> dict:
    """
    Compute probabilities of home win, draw, and away win using independent Poisson goals.

    We approximate by summing probabilities of all scorelines from 0..max_goals for each team.
    """
    lambda_home, lambda_away = expected_goals(home_team, away_team, neutral=neutral)

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    # sum over possible scorelines
    for i in range(0, max_goals + 1):
        for j in range(0, max_goals + 1):
            p_ij = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            if i > j:
                p_home += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away += p_ij

    # small probability mass might be in scorelines beyond max_goals -> renormalise
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "p_home_win": p_home,
        "p_draw": p_draw,
        "p_away_win": p_away,
    }
