# scripts/test_match_prediction.py

import os
import sys

# ensure project root is on the path (same trick as before)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.match_prediction import expected_goals, match_outcome_probabilities


def main():
    home_team = "Argentina"
    away_team = "France"

    lam_home, lam_away = expected_goals(home_team, away_team, neutral=True)
    print(f"Expected goals {home_team} (home): {lam_home:.2f}")
    print(f"Expected goals {away_team} (away): {lam_away:.2f}")

    probs = match_outcome_probabilities(home_team, away_team, neutral=True)
    print("\nMatch outcome probabilities (neutral ground):")
    print(f"P({home_team} win): {probs['p_home_win']:.3f}")
    print(f"P(draw):          {probs['p_draw']:.3f}")
    print(f"P({away_team} win): {probs['p_away_win']:.3f}")


if __name__ == "__main__":
    main()
