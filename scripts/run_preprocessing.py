# scripts/run_preprocessing.py

import os
import sys

# 🔧 Add the project root to Python path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loading import load_results, load_former_names
from src.preprocessing import preprocess_matches, save_processed_matches


def main():
    results = load_results()
    former = load_former_names()

    processed = preprocess_matches(results, former)
    print(processed.head())  # just to check it worked

    save_processed_matches(processed)


if __name__ == "__main__":
    main()
