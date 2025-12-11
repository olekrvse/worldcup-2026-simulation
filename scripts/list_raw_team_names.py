import os
import sys
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loading import load_results, load_former_names
from src.preprocessing import apply_former_name_mapping


def main():
    # Load raw results.csv
    df = load_results()

    # Load former_names.csv
    former = load_former_names()

    # Apply historical name unification
    df = apply_former_name_mapping(df, former)

    # Extract all unique names (home or away)
    teams = sorted(set(df["home_team"]).union(df["away_team"]))

    print(len(teams))
    print(teams)


if __name__ == "__main__":
    main()
