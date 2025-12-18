# README.md (Complete GitHub Version)

# **World Cup 2026 Prediction Model (Poisson + Monte Carlo Simulation)**

*A full probabilistic forecast of the FIFA World Cup 2026, based on real match data (2018–2025)*

---

## **Overview**

This project builds a **complete end-to-end football forecasting pipeline** for the FIFA World Cup 2026, using:

* **Match-level data (2018–2025)** from Kaggle’s International Football Results dataset
* **Team-strength estimation** via a Poisson regression model
* **Monte Carlo simulation** of

  * All group-stage matches
  * Group standings & advancement
  * Knockout bracket generation
  * R32 → R16 → QF → SF → Final
* **Tournament-level probabilities**, including

  * Expected points, goals, goal difference
  * Probability to finish 1st–4th in group
  * Probability to reach R16, QF, SF, Final
  * Probability of becoming **World Champion**

The final output resembles the structure of professional forecasts (e.g., FiveThirtyEight or Opta).

**This model is not intended to predict exact results, but to build a transparent, replicable baseline for team-level strengths and tournament outcomes.**

---

## **Repository Structure**

```
coding_project/
├── data/
│   ├── fixtures/                # WC 2026 custom group-stage fixtures
│   └── processed/               # Model outputs & simulation results
│
├── notebooks/
│   └── wc2026_analysis.ipynb    # Visualisation & interpretation notebook
│
├── scripts/
│   ├── run_preprocessing.py     # Build cleaned match dataset
│   ├── fit_poisson_model.py     # Estimate attack/defence parameters
│   ├── evaluate_group_stage.py  # Match probabilities for group stage
│   ├── simulate_group_stage.py  # Group-stage Monte Carlo simulation
│   ├── simulate_full_tournament.py # Full tournament simulation engine
│   ├── test_match_prediction.py # Sanity checks for match predictions
│   └── list_raw_team_names.py   # Debug helper for team-name alignment
│
├── src/
│   ├── config.py                # Centralised path configuration
│   ├── data_loading.py          # Load raw & processed data
│   ├── preprocessing.py         # Cleaning & normalization logic
│   ├── poisson_model.py         # Poisson regression model
│   ├── match_prediction.py      # Expected goals & W/D/L probabilities
│   └── fixtures_wc2026.py       # WC 2026 fixture definitions
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## **Methodology**

### **1. Data Selection**

We use only **2018–2025** men’s international A-team matches.
This range avoids data contamination from:

* Old teams or outdated strengths
* COVID-era disruptions (but enough matches remain post-2021)
* Legacy teams that no longer exist

Name normalization is handled through `former_names.csv` + custom mapping.

---

### **2. Modelling Approach**

#### **Poisson Regression**

For each match ( i ), home and away goals are modeled as:

[
\lambda_{home} = \exp(\alpha_{home} - \delta_{away} + \beta \cdot \text{home_advantage})
]
[
\lambda_{away} = \exp(\alpha_{away} - \delta_{home})
]

Where:

* ( \alpha ) = attacking strength
* ( \delta ) = defensive strength
* ( \beta ) = home advantage parameter
* Neutral venues set ( \text{home_advantage} = 0 )

Outputs are stored in:

```
data/processed/team_params_poisson.csv
```

---

### **3. Match Prediction Engine**

Expected goals:

```
lambda_home, lambda_away = expected_goals(home_team, away_team, neutral=True)
```

Probabilities computed via joint Poisson:

* P(home win)
* P(draw)
* P(away win)

---

### **4. Monte Carlo Simulation**

We simulate **N = 20,000+ tournaments**.
Each simulation:

1. Computes all group-stage match results
2. Builds final standings with:

   * Points
   * Goal difference
   * Goals scored
3. Applies FIFA tiebreakers
4. Randomly produces the Round-of-32 bracket
5. Simulates all knockout matches
6. Records the round reached by every team

Final outputs stored in:

```
data/processed/wc2026_group_stage_match_probs.csv
data/processed/wc2026_group_stage_simulation_summary.csv
data/processed/wc2026_full_tournament_simulation_summary.csv
```

---

## **Key Results (Selected)**

| Team      | P(Winner)  | P(Final) | P(SF)  |
| --------- | ---------- | -------- | ------ |
| Brazil    | **13.37%** | 8.06%    | 11.40% |
| Spain     | **11.29%** | 7.54%    | 11.01% |
| England   | **9.97%**  | 6.70%    | 11.26% |
| Argentina | **9.53%**  | 7.29%    | 10.89% |

Mid-tier dark horses:

* Netherlands, Denmark, Croatia, Morocco, Uruguay
* All have realistic paths to QF/SF
* Tournament winner probabilities ~2–5%

Outsiders such as New Zealand, Bolivia, Jamaica, Haiti have very low advancement and negligible deep-run probability.

Complete visualization & analysis can be found in the notebook:

```
notebooks/wc2026_analysis.ipynb
```

---

## **How to Reproduce the Results**

### **1. Create virtual environment**

```
python3 -m venv .venv
source .venv/bin/activate
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Run pipeline**

```
python scripts/run_preprocessing.py
python scripts/train_poisson_model.py
python scripts/evaluate_group_stage.py
python scripts/simulate_full_tournament.py
```

### **4. Open notebook**

```
jupyter notebook notebooks/wc2026_analysis.ipynb
```

---

## **Limitations & Extensions**

### Limitations

* Poisson assumes scoring independence
* No player-level or squad-level modelling
* No fatigue, travel, or schedule effects
* Knockout bracket is assumed random unless FIFA final rules are hard-coded
* No pre-match Elo or SPI ratings (deliberate simplification)

### Future Extensions

* Weighted matches (recent matches higher weight)
* Incorporate expected goals (xG) data
* Team-level time-decay
* Player availability modelling
* Bayesian hierarchical Poisson model
* Add FIFA's exact KO bracket logic (incl. 3rd-place seeding rules)

---

## **Author**

This project was developed as a compact but robust football analytics portfolio project.
It demonstrates:

* Data engineering
* Statistical modelling
* Simulation
* Visual analysis
* Clear project structuring

Perfect for showcasing applied analytics and machine learning for sports forecasting.

If you find issues or want extensions, feel free to open a pull request.

---

# **License**

MIT License.

---

