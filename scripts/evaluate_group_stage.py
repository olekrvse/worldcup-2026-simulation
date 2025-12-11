# scripts/evaluate_group_stage.py

import os
import sys
import pandas as pd

# ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fixtures_wc2026 import load_group_stage_fixtures
from src.match_prediction import match_outcome_probabilities
from src.poisson_model import load_processed_matches, extract_team_parameters, fit_poisson_model


def main():
    # Load fixtures
    fixtures = load_group_stage_fixtures()

    # Optional: verify that all teams in fixtures exist in the model
    # (Prevents nasty surprises)
    matches = load_processed_matches()
    result = fit_poisson_model(matches)
    team_params = extract_team_parameters(result)
    model_teams = set(team_params["team"].unique())

    fixture_teams = set(fixtures["home_team"]).union(set(fixtures["away_team"]))
    missing = fixture_teams - model_teams
    if missing:
        raise ValueError(f"The following teams are not in the model parameters: {missing}")

    rows = []
    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("neutral", True))

        probs = match_outcome_probabilities(home, away, neutral=neutral)
        rows.append(
            {
                "date": row["date"],
                "group": row["group"],
                "home_team": home,
                "away_team": away,
                "lambda_home": probs["lambda_home"],
                "lambda_away": probs["lambda_away"],
                "p_home_win": probs["p_home_win"],
                "p_draw": probs["p_draw"],
                "p_away_win": probs["p_away_win"],
            }
        )

    df_probs = pd.DataFrame(rows)
    print(df_probs.head())

    # Save for analysis / plotting
    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "wc2026_group_stage_match_probs.csv")
    df_probs.to_csv(out_path, index=False)
    print(f"\nSaved match probabilities to {out_path}")


if __name__ == "__main__":
    main()
