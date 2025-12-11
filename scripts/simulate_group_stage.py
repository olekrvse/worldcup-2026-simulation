# scripts/simulate_group_stage.py

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fixtures_wc2026 import load_group_stage_fixtures
from src.match_prediction import expected_goals


def simulate_group_stage(n_sim: int = 10_000, random_seed: int = 42) -> pd.DataFrame:
    """
    Run a Monte Carlo simulation of the World Cup 2026 group stage.

    For each simulation:
      - Sample scorelines for every match from independent Poissons
      - Compute points, goals for/against per team
      - Rank teams in each group by: points, goal difference, goals scored, name
      - Record finishing positions

    Returns a DataFrame with, for each team:
      - expected points, goal difference, goals scored
      - probabilities of finishing 1st, 2nd, 3rd, 4th
      - probability of advancing (1st or 2nd)
    """
    fixtures = load_group_stage_fixtures()

    # All teams and group membership
    all_teams = sorted(set(fixtures["home_team"]).union(fixtures["away_team"]))
    groups = {
        group_id: sorted(set(sub["home_team"]).union(sub["away_team"]))
        for group_id, sub in fixtures.groupby("group")
    }

    # Global accumulators across all simulations
    agg = {
        team: {
            "group": None,
            "sum_points": 0.0,
            "sum_gd": 0.0,
            "sum_gf": 0.0,
            "count_first": 0,
            "count_second": 0,
            "count_third": 0,
            "count_fourth": 0,
        }
        for team in all_teams
    }
    # Set group info per team
    for group_id, teams in groups.items():
        for t in teams:
            agg[t]["group"] = group_id

    rng = np.random.default_rng(random_seed)

    for sim in range(n_sim):
        # Per-simulation stats
        points = {team: 0 for team in all_teams}
        gf = {team: 0 for team in all_teams}
        ga = {team: 0 for team in all_teams}

        # Simulate every match once
        for _, row in fixtures.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            neutral = bool(row["neutral"])

            lam_h, lam_a = expected_goals(home, away, neutral=neutral)

            # Sample goals from Poisson distributions
            goals_home = rng.poisson(lam_h)
            goals_away = rng.poisson(lam_a)

            # Update goals for/against
            gf[home] += goals_home
            ga[home] += goals_away
            gf[away] += goals_away
            ga[away] += goals_home

            # Assign points
            if goals_home > goals_away:
                points[home] += 3
            elif goals_home < goals_away:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

        # Rank teams in each group and record positions
        for group_id, teams in groups.items():
            sorted_teams = sorted(
                teams,
                key=lambda t: (points[t], gf[t] - ga[t], gf[t], t),
                reverse=True,
            )
            for idx, team in enumerate(sorted_teams):
                pos = idx + 1
                if pos == 1:
                    agg[team]["count_first"] += 1
                elif pos == 2:
                    agg[team]["count_second"] += 1
                elif pos == 3:
                    agg[team]["count_third"] += 1
                elif pos == 4:
                    agg[team]["count_fourth"] += 1

        # Accumulate expectations
        for t in all_teams:
            agg[t]["sum_points"] += points[t]
            agg[t]["sum_gd"] += (gf[t] - ga[t])
            agg[t]["sum_gf"] += gf[t]

    # Build summary DataFrame
    rows = []
    n = float(n_sim)
    for team, stats in agg.items():
        rows.append(
            {
                "team": team,
                "group": stats["group"],
                "exp_points": stats["sum_points"] / n,
                "exp_gd": stats["sum_gd"] / n,
                "exp_gf": stats["sum_gf"] / n,
                "prob_1st": stats["count_first"] / n,
                "prob_2nd": stats["count_second"] / n,
                "prob_3rd": stats["count_third"] / n,
                "prob_4th": stats["count_fourth"] / n,
                "prob_advance": (stats["count_first"] + stats["count_second"]) / n,
            }
        )

    df_summary = pd.DataFrame(rows)
    return df_summary


def main():
    df_summary = simulate_group_stage(n_sim=10_000, random_seed=42)

    # Sort by group then by probability to advance (descending)
    df_sorted = df_summary.sort_values(
        ["group", "prob_advance", "exp_points"],
        ascending=[True, False, False],
    )

    print(df_sorted.head(24))

    out_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "wc2026_group_stage_simulation_summary.csv",
    )
    df_sorted.to_csv(out_path, index=False)
    print(f"\nSaved simulation summary to {out_path}")


if __name__ == "__main__":
    main()
