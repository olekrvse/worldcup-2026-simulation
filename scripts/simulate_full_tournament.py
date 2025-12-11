# scripts/simulate_full_tournament.py

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


def simulate_group_stage_once(fixtures: pd.DataFrame, rng: np.random.Generator):
    """
    Simulate one full group stage.

    Returns:
      - group_table: dict[group_id -> list of (team, points, gd, gf)]
      - all_points: dict[team -> points]
      - all_gd: dict[team -> goal difference]
      - all_gf: dict[team -> goals for]
    """
    teams = sorted(set(fixtures["home_team"]).union(fixtures["away_team"]))
    points = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}

    # Simulate all group matches
    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row["neutral"])

        lam_h, lam_a = expected_goals(home, away, neutral=neutral)

        goals_home = rng.poisson(lam_h)
        goals_away = rng.poisson(lam_a)

        gf[home] += goals_home
        ga[home] += goals_away
        gf[away] += goals_away
        ga[away] += goals_home

        if goals_home > goals_away:
            points[home] += 3
        elif goals_home < goals_away:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    # Build per-group ranking
    group_table = {}
    for group_id, sub in fixtures.groupby("group"):
        group_teams = sorted(set(sub["home_team"]).union(sub["away_team"]))
        ranked = sorted(
            group_teams,
            key=lambda t: (points[t], gf[t] - ga[t], gf[t], t),
            reverse=True,
        )
        table_rows = [(t, points[t], gf[t] - ga[t], gf[t]) for t in ranked]
        group_table[group_id] = table_rows

    return group_table, points, gf, ga


def simulate_knockout_once(qualifiers: list[str], rng: np.random.Generator):
    """
    Simulate a full 32-team knockout bracket.

    qualifiers: list of 32 team names.
    Returns: dict[team -> furthest round reached]
      R32, R16, QF, SF, F, W (winner)
    """
    # Start with a random bracket for R32
    teams_r32 = qualifiers.copy()
    rng.shuffle(teams_r32)

    round_reached = {t: "Group" for t in qualifiers}

    def play_round(team_list, round_label):
        """
        Play one knockout round.
        team_list: list of teams (even length)
        round_label: label to assign to losers of this round.
        Returns: winners list.
        """
        winners = []
        # Pair teams sequentially
        for i in range(0, len(team_list), 2):
            t1 = team_list[i]
            t2 = team_list[i + 1]

            # Knockout matches are considered neutral
            lam1, lam2 = expected_goals(t1, t2, neutral=True)
            g1 = rng.poisson(lam1)
            g2 = rng.poisson(lam2)

            if g1 > g2:
                winners.append(t1)
                round_reached[t2] = round_label
            elif g2 > g1:
                winners.append(t2)
                round_reached[t1] = round_label
            else:
                # Decide on penalties - simple 50/50
                if rng.random() < 0.5:
                    winners.append(t1)
                    round_reached[t2] = round_label
                else:
                    winners.append(t2)
                    round_reached[t1] = round_label

        return winners

    # Round of 32
    teams_r16 = play_round(teams_r32, "R32")
    # Round of 16
    teams_qf = play_round(teams_r16, "R16")
    # Quarterfinals
    teams_sf = play_round(teams_qf, "QF")
    # Semifinals
    teams_f = play_round(teams_sf, "SF")
    # Final
    # teams_f has length 2
    lam1, lam2 = expected_goals(teams_f[0], teams_f[1], neutral=True)
    g1 = rng.poisson(lam1)
    g2 = rng.poisson(lam2)
    if g1 > g2 or (g1 == g2 and rng.random() < 0.5):
        winner = teams_f[0]
        loser = teams_f[1]
    else:
        winner = teams_f[1]
        loser = teams_f[0]

    round_reached[loser] = "F"
    round_reached[winner] = "W"

    return round_reached


def simulate_full_tournament(n_sim: int = 10_000, random_seed: int = 123) -> pd.DataFrame:
    """
    Simulate the full World Cup 2026 tournament (group + simplified knockout)
    n_sim times.

    Returns a DataFrame with per-team:
      - group
      - expected group points, gd, gf
      - prob_1st, prob_2nd, prob_3rd, prob_4th
      - prob_qualify (enter R32)
      - prob_R16, prob_QF, prob_SF, prob_F, prob_W (champion)
    """
    fixtures = load_group_stage_fixtures()
    groups = {
        group_id: sorted(set(sub["home_team"]).union(sub["away_team"]))
        for group_id, sub in fixtures.groupby("group")
    }
    all_teams = sorted(set(fixtures["home_team"]).union(fixtures["away_team"]))

    rng = np.random.default_rng(random_seed)

    # Aggregators
    agg = {
        t: {
            "group": None,
            "sum_points": 0.0,
            "sum_gd": 0.0,
            "sum_gf": 0.0,
            "count_1st": 0,
            "count_2nd": 0,
            "count_3rd": 0,
            "count_4th": 0,
            "count_qual": 0,  # reached R32
            "count_R16": 0,
            "count_QF": 0,
            "count_SF": 0,
            "count_F": 0,
            "count_W": 0,
        }
        for t in all_teams
    }
    # Store group for each team
    for g, teams in groups.items():
        for t in teams:
            agg[t]["group"] = g

    for sim in range(n_sim):
        # 1) Group stage
        group_table, points, gf, ga = simulate_group_stage_once(fixtures, rng)

        # Record group stats
        for g, rows in group_table.items():
            # rows is list of (team, pts, gd, gf) in order
            for pos_idx, (team, pts, gd, gf_t) in enumerate(rows):
                pos = pos_idx + 1
                agg[team]["sum_points"] += pts
                agg[team]["sum_gd"] += gd
                agg[team]["sum_gf"] += gf_t
                if pos == 1:
                    agg[team]["count_1st"] += 1
                elif pos == 2:
                    agg[team]["count_2nd"] += 1
                elif pos == 3:
                    agg[team]["count_3rd"] += 1
                elif pos == 4:
                    agg[team]["count_4th"] += 1

        # 2) Determine qualifiers (simple rule: top 2 + best 8 third-place teams)
        qualifiers = []

        third_place_candidates = []
        for g, rows in group_table.items():
            # rows already sorted
            first_team, first_pts, _, _ = rows[0]
            second_team, second_pts, _, _ = rows[1]
            third_team, third_pts, third_gd, third_gf = rows[2]

            qualifiers.append(first_team)
            qualifiers.append(second_team)

            # Collect third-place teams for ranking across groups
            third_place_candidates.append(
                (third_team, g, third_pts, third_gd, third_gf)
            )

        # Rank third-place teams across all groups:
        # by points, gd, gf, then random tie-breaker
        rng.shuffle(third_place_candidates)  # add randomness in exact ties
        third_place_candidates.sort(
            key=lambda x: (x[2], x[3], x[4]),  # pts, gd, gf
            reverse=True,
        )
        best_thirds = [t[0] for t in third_place_candidates[:8]]

        qualifiers.extend(best_thirds)

        # Mark qualification
        for team in qualifiers:
            agg[team]["count_qual"] += 1

        # 3) Knockout simulation for those 32 teams
        round_reached = simulate_knockout_once(qualifiers, rng)

        for team, r in round_reached.items():
            if r == "R16":
                agg[team]["count_R16"] += 1
            elif r == "QF":
                agg[team]["count_QF"] += 1
            elif r == "SF":
                agg[team]["count_SF"] += 1
            elif r == "F":
                agg[team]["count_F"] += 1
            elif r == "W":
                agg[team]["count_W"] += 1
            # Teams eliminated in R32 have r == "R32"; we don't store separately here.

    # 4) Build final DataFrame
    n = float(n_sim)
    rows = []
    for t, s in agg.items():
        rows.append(
            {
                "team": t,
                "group": s["group"],
                "exp_points": s["sum_points"] / n,
                "exp_gd": s["sum_gd"] / n,
                "exp_gf": s["sum_gf"] / n,
                "prob_1st": s["count_1st"] / n,
                "prob_2nd": s["count_2nd"] / n,
                "prob_3rd": s["count_3rd"] / n,
                "prob_4th": s["count_4th"] / n,
                "prob_qual": s["count_qual"] / n,
                "prob_R16": s["count_R16"] / n,
                "prob_QF": s["count_QF"] / n,
                "prob_SF": s["count_SF"] / n,
                "prob_F": s["count_F"] / n,
                "prob_W": s["count_W"] / n,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main():
    df = simulate_full_tournament(n_sim=10_000, random_seed=123)

    df_sorted = df.sort_values(
        ["prob_W", "prob_F", "prob_SF", "prob_QF"],
        ascending=[False, False, False, False],
    )

    print(df_sorted.head(20))

    out_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "wc2026_full_tournament_simulation_summary.csv",
    )
    df_sorted.to_csv(out_path, index=False)
    print(f"\nSaved full tournament simulation summary to {out_path}")


if __name__ == "__main__":
    main()
