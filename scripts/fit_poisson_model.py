# scripts/fit_poisson_model.py

import os
import sys

# 🔧 Make sure the project root is on the path (same trick as before)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.poisson_model import fit_and_save_model


if __name__ == "__main__":
    fit_and_save_model()
