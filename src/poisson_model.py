# src/poisson_model.py

import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import DATA_PROCESSED_DIR


def load_processed_matches(filename: str = "matches_2018_2025.csv") -> pd.DataFrame:
    """Load the cleaned match data used for modelling."""
    path = os.path.join(DATA_PROCESSED_DIR, filename)
    df = pd.read_csv(path)
    return df


def build_team_goal_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Turn each match into two rows:
    - one for home team goals
    - one for away team goals

    Columns: team, opponent, home (0/1), goals
    """
    rows = []

    for _, row in matches.iterrows():
        # Home team row
        rows.append(
            {
                "team": row["home_team"],
                "opponent": row["away_team"],
                "home": 1,
                "goals": row["home_score"],
            }
        )
        # Away team row
        rows.append(
            {
                "team": row["away_team"],
                "opponent": row["home_team"],
                "home": 0,
                "goals": row["away_score"],
            }
        )

    return pd.DataFrame(rows)


def fit_poisson_model(matches: pd.DataFrame):
    """
    Fit a Poisson regression model:
        goals ~ home + C(team) + C(opponent)

    This gives:
    - attack contribution for each team (C(team))
    - defensive 'leakiness' for each team (C(opponent))
    - home advantage (home)
    """
    df_goals = build_team_goal_dataset(matches)

    model = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=df_goals,
        family=sm.families.Poisson(),
    )
    result = model.fit()
    return result


def extract_team_parameters(result) -> pd.DataFrame:
    """
    From the fitted model, extract per-team attack and defence parameters.

    log(λ_team_vs_opponent) ≈ Intercept + home*β_home + attack_team + defence_opponent

    Where:
    - attack_team  comes from C(team)[T.team]
    - defence_team comes from C(opponent)[T.team]
    The reference team (alphabetically first) has 0 for attack/defence by construction.
    """
    params = result.params

    intercept = params.get("Intercept", 0.0)
    home_adv = params.get("home", 0.0)

    # All parameter names for team attack and opponent defence
    attack_coefs = {k: v for k, v in params.items() if k.startswith("C(team)[T.")}
    defence_coefs = {k: v for k, v in params.items() if k.startswith("C(opponent)[T.")}

    # Get team names from the coefficient labels
    def _extract_name(prefix: str, full: str) -> str:
        # e.g. "C(team)[T.Brazil]" -> "Brazil"
        return full.replace(prefix, "").rstrip("]")

    attack_teams = { _extract_name("C(team)[T.", k): v for k, v in attack_coefs.items() }
    defence_teams = { _extract_name("C(opponent)[T.", k): v for k, v in defence_coefs.items() }

    # Collect all teams that appear in either side
    all_teams = sorted(set(list(attack_teams.keys()) + list(defence_teams.keys())))

    rows = []
    for team in all_teams:
        att = attack_teams.get(team, 0.0)   # reference team -> 0
        defe = defence_teams.get(team, 0.0)

        rows.append(
            {
                "team": team,
                "attack": att,
                "defence": defe,
                "intercept": intercept,
                "home_advantage": home_adv,
            }
        )

    df_params = pd.DataFrame(rows)
    return df_params


def fit_and_save_model():
    """
    Convenience function:
    - load processed matches
    - fit Poisson model
    - extract per-team parameters
    - save to CSV
    """
    matches = load_processed_matches()
    print(f"Loaded {len(matches)} matches for 2018–2025.")

    result = fit_poisson_model(matches)
    print(result.summary())

    team_params = extract_team_parameters(result)

    out_path = os.path.join(DATA_PROCESSED_DIR, "team_params_poisson.csv")
    team_params.to_csv(out_path, index=False)
    print(f"\nSaved team parameters to {out_path}")
