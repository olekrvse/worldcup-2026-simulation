# FIFA World Cup 2026 – Poisson Model & Monte Carlo Simulation

This notebook analyses a full-tournament prediction model for the 2026 FIFA World Cup, based on:

- Match data from international fixtures (2018–2025)
- A Poisson regression model estimating team attack/defence strength and home advantage
- Monte Carlo simulations of:
  - The group stage (match-by-match Poisson score sampling)
  - A 32-team knockout bracket (Round of 32 → Final)

**Key outputs:**

- Group-stage expectations (points, qualification odds, finishing positions)
- Knockout progression probabilities (R16, QF, SF, Final, Champion)
- Visualisations and short interpretations for portfolio / GitHub / LinkedIn.



```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Display options
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

PROJECT_ROOT = "/Users/Ole_Kruse/Desktop/coding_project"
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


PROJECT_ROOT, DATA_PROCESSED_DIR

```




    ('/Users/Ole_Kruse/Desktop/coding_project',
     '/Users/Ole_Kruse/Desktop/coding_project/data/processed')




```python
# Group-stage match-level probabilities
group_match_probs_path = os.path.join(
    DATA_PROCESSED_DIR, "wc2026_group_stage_match_probs.csv"
)
group_match_probs = pd.read_csv(group_match_probs_path, parse_dates=["date"])

# Full-tournament simulation summary (group + knockout)
full_tournament_path = os.path.join(
    DATA_PROCESSED_DIR, "wc2026_full_tournament_simulation_summary.csv"
)
tournament_summary = pd.read_csv(full_tournament_path)

group_match_probs.head(), tournament_summary.head()

```




    (        date group      home_team     away_team  lambda_home  lambda_away  \
     0 2026-06-11     A         Mexico  South Africa       1.0773       0.6939   
     1 2026-06-11     A    South Korea       Denmark       0.6947       1.4559   
     2 2026-06-12     B         Canada         Italy       0.7559       1.3558   
     3 2026-06-12     D  United States      Paraguay       1.0179       0.8571   
     4 2026-06-13     C          Haiti      Scotland       0.8035       1.9068   
     
        p_home_win  p_draw  p_away_win  
     0      0.4431  0.3231      0.2338  
     1      0.1780  0.2675      0.5545  
     2      0.2086  0.2807      0.5106  
     3      0.3835  0.3193      0.2972  
     4      0.1512  0.2148      0.6340  ,
             team group  exp_points  exp_gd  exp_gf  prob_1st  prob_2nd  prob_3rd  \
     0     Brazil     C      7.1574  5.5222  6.8905    0.7423    0.2082    0.0450   
     1      Spain     H      6.9762  4.6673  6.2500    0.6891    0.2634    0.0391   
     2    England     L      6.9899  4.5885  6.2457    0.6974    0.2410    0.0521   
     3  Argentina     J      6.7885  4.1966  5.9084    0.6729    0.2327    0.0816   
     4   Portugal     K      6.5593  4.2909  6.2832    0.5452    0.3737    0.0719   
     
        prob_4th  prob_qual  prob_R16  prob_QF  prob_SF  prob_F  prob_W  
     0    0.0045     0.9921    0.2253   0.1631   0.1140  0.0806  0.1337  
     1    0.0084     0.9852    0.2200   0.1729   0.1101  0.0754  0.1129  
     2    0.0095     0.9834    0.2239   0.1651   0.1126  0.0670  0.0997  
     3    0.0128     0.9770    0.2293   0.1553   0.1089  0.0729  0.0953  
     4    0.0092     0.9803    0.2431   0.1590   0.0973  0.0631  0.0711  )



## 1. Data overview

We first inspect the two main outputs:

1. `wc2026_group_stage_match_probs.csv`
   - One row per group-stage match
   - Expected goals for each team (`lambda_home`, `lambda_away`)
   - Win/draw/loss probabilities (`p_home_win`, `p_draw`, `p_away_win`)

2. `wc2026_full_tournament_simulation_summary.csv`
   - One row per team
   - Group-stage expectations (points, goal difference, goals scored, position probabilities)
   - Knockout probabilities (reaching R32, R16, QF, SF, Final, Champion)



```python
print("Group match probabilities:")
display(group_match_probs.describe())

print("\nFull tournament summary:")
display(tournament_summary.describe())

print("\nColumns in tournament_summary:")
print(tournament_summary.columns.tolist())

```

    Group match probabilities:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>lambda_home</th>
      <th>lambda_away</th>
      <th>p_home_win</th>
      <th>p_draw</th>
      <th>p_away_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>72</td>
      <td>72.0000</td>
      <td>72.0000</td>
      <td>72.0000</td>
      <td>72.0000</td>
      <td>72.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2026-06-20 03:20:00</td>
      <td>1.2994</td>
      <td>1.0567</td>
      <td>0.4247</td>
      <td>0.2413</td>
      <td>0.3340</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2026-06-11 00:00:00</td>
      <td>0.3668</td>
      <td>0.3540</td>
      <td>0.0343</td>
      <td>0.0516</td>
      <td>0.0138</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2026-06-16 00:00:00</td>
      <td>0.7684</td>
      <td>0.6506</td>
      <td>0.2080</td>
      <td>0.1913</td>
      <td>0.1564</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2026-06-20 12:00:00</td>
      <td>1.1454</td>
      <td>0.9625</td>
      <td>0.3826</td>
      <td>0.2579</td>
      <td>0.3187</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2026-06-24 06:00:00</td>
      <td>1.6093</td>
      <td>1.3013</td>
      <td>0.5891</td>
      <td>0.2938</td>
      <td>0.4979</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2026-06-27 00:00:00</td>
      <td>3.6269</td>
      <td>2.6954</td>
      <td>0.9346</td>
      <td>0.3528</td>
      <td>0.8533</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>0.7111</td>
      <td>0.5700</td>
      <td>0.2477</td>
      <td>0.0715</td>
      <td>0.2301</td>
    </tr>
  </tbody>
</table>
</div>


    
    Full tournament summary:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
      <th>prob_R16</th>
      <th>prob_QF</th>
      <th>prob_SF</th>
      <th>prob_F</th>
      <th>prob_W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
      <td>48.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.1386</td>
      <td>-0.0000</td>
      <td>3.5348</td>
      <td>0.2500</td>
      <td>0.2500</td>
      <td>0.2500</td>
      <td>0.2500</td>
      <td>0.6667</td>
      <td>0.1667</td>
      <td>0.0833</td>
      <td>0.0417</td>
      <td>0.0208</td>
      <td>0.0208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.7655</td>
      <td>2.9660</td>
      <td>1.5243</td>
      <td>0.2241</td>
      <td>0.1265</td>
      <td>0.1271</td>
      <td>0.2482</td>
      <td>0.2841</td>
      <td>0.0787</td>
      <td>0.0551</td>
      <td>0.0365</td>
      <td>0.0234</td>
      <td>0.0328</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0594</td>
      <td>-5.9250</td>
      <td>1.2711</td>
      <td>0.0040</td>
      <td>0.0270</td>
      <td>0.0391</td>
      <td>0.0045</td>
      <td>0.0912</td>
      <td>0.0119</td>
      <td>0.0018</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.7704</td>
      <td>-1.7537</td>
      <td>2.3347</td>
      <td>0.0558</td>
      <td>0.1803</td>
      <td>0.1454</td>
      <td>0.0481</td>
      <td>0.4731</td>
      <td>0.1052</td>
      <td>0.0317</td>
      <td>0.0089</td>
      <td>0.0023</td>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.1440</td>
      <td>0.0369</td>
      <td>3.3224</td>
      <td>0.1859</td>
      <td>0.2588</td>
      <td>0.2482</td>
      <td>0.1767</td>
      <td>0.7289</td>
      <td>0.1908</td>
      <td>0.0770</td>
      <td>0.0311</td>
      <td>0.0113</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.4756</td>
      <td>2.0689</td>
      <td>4.5456</td>
      <td>0.3960</td>
      <td>0.3257</td>
      <td>0.3219</td>
      <td>0.3492</td>
      <td>0.9050</td>
      <td>0.2309</td>
      <td>0.1399</td>
      <td>0.0743</td>
      <td>0.0330</td>
      <td>0.0263</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.1574</td>
      <td>5.5222</td>
      <td>6.8905</td>
      <td>0.7423</td>
      <td>0.5402</td>
      <td>0.5818</td>
      <td>0.8358</td>
      <td>0.9921</td>
      <td>0.2615</td>
      <td>0.1729</td>
      <td>0.1140</td>
      <td>0.0806</td>
      <td>0.1337</td>
    </tr>
  </tbody>
</table>
</div>


    
    Columns in tournament_summary:
    ['team', 'group', 'exp_points', 'exp_gd', 'exp_gf', 'prob_1st', 'prob_2nd', 'prob_3rd', 'prob_4th', 'prob_qual', 'prob_R16', 'prob_QF', 'prob_SF', 'prob_F', 'prob_W']


## 2. Tournament favourites – Win probabilities

We start with a simple ranking of teams by the probability of winning the World Cup (`prob_W`).

This gives a clear picture of which teams the model considers top contenders and how concentrated the title chances are.



```python
# Sort teams by probability of winning the World Cup
favorites = tournament_summary.sort_values("prob_W", ascending=False).reset_index(drop=True)

# Take top 15 for a compact view
top_n = 15
favorites_top = favorites.head(top_n)

display(favorites_top[["team", "group", "prob_W", "prob_F", "prob_SF", "prob_QF"]])

# Bar plot of winner probabilities
plt.figure(figsize=(10, 6))
plt.barh(favorites_top["team"][::-1], favorites_top["prob_W"][::-1])
plt.xlabel("Probability of winning the World Cup")
plt.title("World Cup 2026 – Title probabilities (Top 15 teams)")
plt.tight_layout()
plt.show()

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>group</th>
      <th>prob_W</th>
      <th>prob_F</th>
      <th>prob_SF</th>
      <th>prob_QF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>C</td>
      <td>0.1337</td>
      <td>0.0806</td>
      <td>0.1140</td>
      <td>0.1631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>H</td>
      <td>0.1129</td>
      <td>0.0754</td>
      <td>0.1101</td>
      <td>0.1729</td>
    </tr>
    <tr>
      <th>2</th>
      <td>England</td>
      <td>L</td>
      <td>0.0997</td>
      <td>0.0670</td>
      <td>0.1126</td>
      <td>0.1651</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Argentina</td>
      <td>J</td>
      <td>0.0953</td>
      <td>0.0729</td>
      <td>0.1089</td>
      <td>0.1553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Portugal</td>
      <td>K</td>
      <td>0.0711</td>
      <td>0.0631</td>
      <td>0.0973</td>
      <td>0.1590</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>I</td>
      <td>0.0676</td>
      <td>0.0614</td>
      <td>0.1035</td>
      <td>0.1549</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Colombia</td>
      <td>K</td>
      <td>0.0530</td>
      <td>0.0496</td>
      <td>0.0906</td>
      <td>0.1566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Belgium</td>
      <td>G</td>
      <td>0.0499</td>
      <td>0.0532</td>
      <td>0.0975</td>
      <td>0.1602</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>F</td>
      <td>0.0429</td>
      <td>0.0432</td>
      <td>0.0799</td>
      <td>0.1454</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Denmark</td>
      <td>A</td>
      <td>0.0313</td>
      <td>0.0372</td>
      <td>0.0751</td>
      <td>0.1433</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Uruguay</td>
      <td>H</td>
      <td>0.0310</td>
      <td>0.0326</td>
      <td>0.0741</td>
      <td>0.1397</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Italy</td>
      <td>B</td>
      <td>0.0279</td>
      <td>0.0375</td>
      <td>0.0773</td>
      <td>0.1405</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Germany</td>
      <td>E</td>
      <td>0.0257</td>
      <td>0.0344</td>
      <td>0.0756</td>
      <td>0.1464</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Switzerland</td>
      <td>B</td>
      <td>0.0212</td>
      <td>0.0277</td>
      <td>0.0644</td>
      <td>0.1255</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morocco</td>
      <td>C</td>
      <td>0.0180</td>
      <td>0.0249</td>
      <td>0.0586</td>
      <td>0.1233</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_6_1.png)
    


### Interpretation

Some points you might mention in your README / LinkedIn post:

- **Top tier:** Brazil, Spain, England, Argentina form the leading cluster with roughly 9–13% title probability each.
- **Next tier:** Portugal, France, Colombia, Belgium sit just below, reflecting strong recent performance but a slightly tougher path.
- **Dark horses:** Netherlands, Denmark, Uruguay, Croatia, Germany and Italy still have non-trivial paths to the trophy, but require multiple favourable results in high-variance knockout matches.
- The model reflects **realistic parity** at the very top: no team has an absurdly high (>30%) title probability, because the knockout bracket is random and strong teams can meet early.


## 3. Group-by-group qualification and finishing probabilities

For each group (A–L), we visualise:

- Probability of finishing **1st**, **2nd**, **3rd**, **4th**
- Probability of **qualifying for the knockout phase** (`prob_qual` – reaching the Round of 32)

This illustrates:
- Which groups are “groups of death”
- Which teams are overwhelming favourites vs balanced groups



```python
def plot_group_overview(group_id: str, df: pd.DataFrame):
    """
    Plot finishing position & qualification probabilities for one group.
    """
    group_df = df[df["group"] == group_id].copy()
    group_df = group_df.sort_values("prob_1st", ascending=False)

    display(group_df[[
        "team", "exp_points", "exp_gd", "exp_gf",
        "prob_1st", "prob_2nd", "prob_3rd", "prob_4th", "prob_qual"
    ]])

    teams = group_df["team"]

    # Stacked bar for positions
    pos_matrix = group_df[["prob_1st", "prob_2nd", "prob_3rd", "prob_4th"]].values

    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(teams))
    labels = ["1st", "2nd", "3rd", "4th"]
    for i, label in enumerate(labels):
        plt.bar(teams, pos_matrix[:, i], bottom=bottom, label=label)
        bottom += pos_matrix[:, i]

    plt.ylabel("Probability")
    plt.title(f"Group {group_id} – Finishing position probabilities")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Position")
    plt.tight_layout()
    plt.show()

    # Qualification probability
    plt.figure(figsize=(8, 4))
    plt.bar(teams, group_df["prob_qual"])
    plt.ylabel("Probability of reaching Round of 32")
    plt.title(f"Group {group_id} – Qualification probabilities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

```


```python
all_groups = sorted(tournament_summary["group"].unique())
all_groups

```




    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']




```python
for g in all_groups:
    print(f"\n=== Group {g} ===")
    plot_group_overview(g, tournament_summary)

```

    
    === Group A ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Denmark</td>
      <td>5.8214</td>
      <td>2.2936</td>
      <td>4.1908</td>
      <td>0.5417</td>
      <td>0.2585</td>
      <td>0.1398</td>
      <td>0.0600</td>
      <td>0.9163</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Mexico</td>
      <td>3.9925</td>
      <td>-0.0818</td>
      <td>2.9235</td>
      <td>0.2081</td>
      <td>0.2916</td>
      <td>0.2730</td>
      <td>0.2273</td>
      <td>0.6988</td>
    </tr>
    <tr>
      <th>33</th>
      <td>South Korea</td>
      <td>3.6305</td>
      <td>-0.5961</td>
      <td>2.7336</td>
      <td>0.1680</td>
      <td>0.2662</td>
      <td>0.2905</td>
      <td>0.2753</td>
      <td>0.6323</td>
    </tr>
    <tr>
      <th>35</th>
      <td>South Africa</td>
      <td>2.7803</td>
      <td>-1.6157</td>
      <td>1.9233</td>
      <td>0.0822</td>
      <td>0.1837</td>
      <td>0.2967</td>
      <td>0.4374</td>
      <td>0.4529</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_2.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_3.png)
    


    
    === Group B ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Italy</td>
      <td>5.7150</td>
      <td>2.4479</td>
      <td>4.8895</td>
      <td>0.4494</td>
      <td>0.3251</td>
      <td>0.1799</td>
      <td>0.0456</td>
      <td>0.9247</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Switzerland</td>
      <td>5.4031</td>
      <td>2.0640</td>
      <td>4.8507</td>
      <td>0.3875</td>
      <td>0.3462</td>
      <td>0.2071</td>
      <td>0.0592</td>
      <td>0.9012</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Canada</td>
      <td>3.8734</td>
      <td>-0.2482</td>
      <td>3.3523</td>
      <td>0.1433</td>
      <td>0.2606</td>
      <td>0.4089</td>
      <td>0.1872</td>
      <td>0.7042</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Qatar</td>
      <td>1.6201</td>
      <td>-4.2637</td>
      <td>2.2043</td>
      <td>0.0198</td>
      <td>0.0681</td>
      <td>0.2041</td>
      <td>0.7080</td>
      <td>0.2073</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_6.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_7.png)
    


    
    === Group C ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>7.1574</td>
      <td>5.5222</td>
      <td>6.8905</td>
      <td>0.7423</td>
      <td>0.2082</td>
      <td>0.0450</td>
      <td>0.0045</td>
      <td>0.9921</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morocco</td>
      <td>5.0331</td>
      <td>1.4579</td>
      <td>3.6994</td>
      <td>0.1928</td>
      <td>0.5133</td>
      <td>0.2483</td>
      <td>0.0456</td>
      <td>0.8980</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Scotland</td>
      <td>3.4046</td>
      <td>-1.0551</td>
      <td>3.0121</td>
      <td>0.0609</td>
      <td>0.2408</td>
      <td>0.5307</td>
      <td>0.1676</td>
      <td>0.6511</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Haiti</td>
      <td>1.1679</td>
      <td>-5.9250</td>
      <td>1.6033</td>
      <td>0.0040</td>
      <td>0.0377</td>
      <td>0.1760</td>
      <td>0.7823</td>
      <td>0.1190</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_10.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_11.png)
    


    
    === Group D ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Australia</td>
      <td>4.3863</td>
      <td>0.4343</td>
      <td>3.2841</td>
      <td>0.2920</td>
      <td>0.2660</td>
      <td>0.2380</td>
      <td>0.2040</td>
      <td>0.7399</td>
    </tr>
    <tr>
      <th>25</th>
      <td>United States</td>
      <td>4.2058</td>
      <td>0.2258</td>
      <td>3.3419</td>
      <td>0.2821</td>
      <td>0.2590</td>
      <td>0.2373</td>
      <td>0.2216</td>
      <td>0.7180</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Turkey</td>
      <td>3.8987</td>
      <td>-0.2526</td>
      <td>3.4183</td>
      <td>0.2363</td>
      <td>0.2353</td>
      <td>0.2482</td>
      <td>0.2802</td>
      <td>0.6587</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Paraguay</td>
      <td>3.6949</td>
      <td>-0.4075</td>
      <td>2.6842</td>
      <td>0.1896</td>
      <td>0.2397</td>
      <td>0.2765</td>
      <td>0.2942</td>
      <td>0.6304</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_14.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_15.png)
    


    
    === Group E ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Germany</td>
      <td>6.1964</td>
      <td>3.7935</td>
      <td>6.1331</td>
      <td>0.5495</td>
      <td>0.2885</td>
      <td>0.1456</td>
      <td>0.0164</td>
      <td>0.9700</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ecuador</td>
      <td>5.2171</td>
      <td>1.6870</td>
      <td>3.9816</td>
      <td>0.2946</td>
      <td>0.3830</td>
      <td>0.2735</td>
      <td>0.0489</td>
      <td>0.8943</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Ivory Coast</td>
      <td>4.1741</td>
      <td>0.1720</td>
      <td>3.2953</td>
      <td>0.1515</td>
      <td>0.3015</td>
      <td>0.4481</td>
      <td>0.0989</td>
      <td>0.7781</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Curaçao</td>
      <td>1.0594</td>
      <td>-5.6525</td>
      <td>1.3733</td>
      <td>0.0044</td>
      <td>0.0270</td>
      <td>0.1328</td>
      <td>0.8358</td>
      <td>0.0912</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_18.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_19.png)
    


    
    === Group F ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>5.5906</td>
      <td>2.0836</td>
      <td>4.7069</td>
      <td>0.5084</td>
      <td>0.2627</td>
      <td>0.1448</td>
      <td>0.0841</td>
      <td>0.8891</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Japan</td>
      <td>4.0539</td>
      <td>-0.1088</td>
      <td>3.4279</td>
      <td>0.2174</td>
      <td>0.2792</td>
      <td>0.2711</td>
      <td>0.2323</td>
      <td>0.6951</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Tunisia</td>
      <td>3.4791</td>
      <td>-0.6971</td>
      <td>2.5127</td>
      <td>0.1451</td>
      <td>0.2466</td>
      <td>0.3013</td>
      <td>0.3070</td>
      <td>0.5913</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Poland</td>
      <td>3.2118</td>
      <td>-1.2777</td>
      <td>2.8517</td>
      <td>0.1291</td>
      <td>0.2115</td>
      <td>0.2828</td>
      <td>0.3766</td>
      <td>0.5361</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_22.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_23.png)
    


    
    === Group G ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Belgium</td>
      <td>6.5253</td>
      <td>3.4945</td>
      <td>5.2097</td>
      <td>0.6588</td>
      <td>0.2297</td>
      <td>0.0890</td>
      <td>0.0225</td>
      <td>0.9652</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Iran</td>
      <td>4.1788</td>
      <td>0.0501</td>
      <td>2.8592</td>
      <td>0.1811</td>
      <td>0.3560</td>
      <td>0.3199</td>
      <td>0.1430</td>
      <td>0.7530</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Egypt</td>
      <td>3.8822</td>
      <td>-0.2860</td>
      <td>2.5464</td>
      <td>0.1398</td>
      <td>0.3211</td>
      <td>0.3533</td>
      <td>0.1858</td>
      <td>0.6912</td>
    </tr>
    <tr>
      <th>41</th>
      <td>New Zealand</td>
      <td>1.7747</td>
      <td>-3.2586</td>
      <td>1.2711</td>
      <td>0.0203</td>
      <td>0.0932</td>
      <td>0.2378</td>
      <td>0.6487</td>
      <td>0.2276</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_26.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_27.png)
    


    
    === Group H ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>6.9762</td>
      <td>4.6673</td>
      <td>6.2500</td>
      <td>0.6891</td>
      <td>0.2634</td>
      <td>0.0391</td>
      <td>0.0084</td>
      <td>0.9852</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Uruguay</td>
      <td>5.4372</td>
      <td>1.8510</td>
      <td>3.9810</td>
      <td>0.2795</td>
      <td>0.5402</td>
      <td>0.1420</td>
      <td>0.0383</td>
      <td>0.9202</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Cape Verde</td>
      <td>2.0826</td>
      <td>-3.3872</td>
      <td>1.6413</td>
      <td>0.0159</td>
      <td>0.0929</td>
      <td>0.3984</td>
      <td>0.4928</td>
      <td>0.2935</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Saudi Arabia</td>
      <td>2.1161</td>
      <td>-3.1311</td>
      <td>1.5981</td>
      <td>0.0155</td>
      <td>0.1035</td>
      <td>0.4205</td>
      <td>0.4605</td>
      <td>0.3101</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_30.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_31.png)
    


    
    === Group I ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>6.4095</td>
      <td>3.5837</td>
      <td>5.6975</td>
      <td>0.6110</td>
      <td>0.2520</td>
      <td>0.1134</td>
      <td>0.0236</td>
      <td>0.9614</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Norway</td>
      <td>4.4444</td>
      <td>0.5927</td>
      <td>4.0823</td>
      <td>0.2039</td>
      <td>0.3476</td>
      <td>0.3279</td>
      <td>0.1206</td>
      <td>0.7981</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Senegal</td>
      <td>4.2149</td>
      <td>0.2227</td>
      <td>3.3029</td>
      <td>0.1677</td>
      <td>0.3351</td>
      <td>0.3645</td>
      <td>0.1327</td>
      <td>0.7671</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Bolivia</td>
      <td>1.5543</td>
      <td>-4.3991</td>
      <td>2.0055</td>
      <td>0.0174</td>
      <td>0.0653</td>
      <td>0.1942</td>
      <td>0.7231</td>
      <td>0.1931</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_34.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_35.png)
    


    
    === Group J ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Argentina</td>
      <td>6.7885</td>
      <td>4.1966</td>
      <td>5.9084</td>
      <td>0.6729</td>
      <td>0.2327</td>
      <td>0.0816</td>
      <td>0.0128</td>
      <td>0.9770</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Austria</td>
      <td>4.5070</td>
      <td>0.6530</td>
      <td>4.0035</td>
      <td>0.1821</td>
      <td>0.3877</td>
      <td>0.3316</td>
      <td>0.0986</td>
      <td>0.8165</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Algeria</td>
      <td>4.1139</td>
      <td>0.0238</td>
      <td>3.8286</td>
      <td>0.1367</td>
      <td>0.3277</td>
      <td>0.3983</td>
      <td>0.1373</td>
      <td>0.7560</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Jordan</td>
      <td>1.3278</td>
      <td>-4.8734</td>
      <td>1.7401</td>
      <td>0.0083</td>
      <td>0.0519</td>
      <td>0.1885</td>
      <td>0.7513</td>
      <td>0.1550</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_38.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_39.png)
    


    
    === Group K ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Portugal</td>
      <td>6.5593</td>
      <td>4.2909</td>
      <td>6.2832</td>
      <td>0.5452</td>
      <td>0.3737</td>
      <td>0.0719</td>
      <td>0.0092</td>
      <td>0.9803</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Colombia</td>
      <td>6.2266</td>
      <td>3.3234</td>
      <td>5.2875</td>
      <td>0.4216</td>
      <td>0.4663</td>
      <td>0.0974</td>
      <td>0.0147</td>
      <td>0.9670</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Uzbekistan</td>
      <td>2.7409</td>
      <td>-2.2442</td>
      <td>2.3061</td>
      <td>0.0288</td>
      <td>0.1289</td>
      <td>0.5818</td>
      <td>0.2605</td>
      <td>0.4767</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Jamaica</td>
      <td>1.2919</td>
      <td>-5.3701</td>
      <td>1.5130</td>
      <td>0.0044</td>
      <td>0.0311</td>
      <td>0.2489</td>
      <td>0.7156</td>
      <td>0.1336</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_42.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_43.png)
    


    
    === Group L ===



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>exp_points</th>
      <th>exp_gd</th>
      <th>exp_gf</th>
      <th>prob_1st</th>
      <th>prob_2nd</th>
      <th>prob_3rd</th>
      <th>prob_4th</th>
      <th>prob_qual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>England</td>
      <td>6.9899</td>
      <td>4.5885</td>
      <td>6.2457</td>
      <td>0.6974</td>
      <td>0.2410</td>
      <td>0.0521</td>
      <td>0.0095</td>
      <td>0.9834</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Croatia</td>
      <td>5.1770</td>
      <td>1.6127</td>
      <td>4.4918</td>
      <td>0.2452</td>
      <td>0.5003</td>
      <td>0.1900</td>
      <td>0.0645</td>
      <td>0.8841</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Ghana</td>
      <td>2.7207</td>
      <td>-2.1678</td>
      <td>2.3442</td>
      <td>0.0406</td>
      <td>0.1702</td>
      <td>0.4491</td>
      <td>0.3401</td>
      <td>0.4623</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Panama</td>
      <td>1.8452</td>
      <td>-4.0334</td>
      <td>1.9913</td>
      <td>0.0168</td>
      <td>0.0885</td>
      <td>0.3088</td>
      <td>0.5859</td>
      <td>0.2523</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_11_46.png)
    



    
![png](wc2026_analysis_files/wc2026_analysis_11_47.png)
    


## 4. Deep-run probabilities (SF, Final, Champion)

Next we focus on the probability that teams reach:

- At least the **Quarterfinals** (`prob_QF`)
- At least the **Semifinals** (`prob_SF`)
- The **Final** (`prob_F`)
- Win the **World Cup** (`prob_W`)

This highlights not only favourites to win, but also teams that are likely to make a deep run.



```python
# Select relevant columns
deep_run = tournament_summary[[
    "team", "group", "prob_QF", "prob_SF", "prob_F", "prob_W"
]].copy()

# Sort by probability of reaching semi-finals as a proxy for deep run strength
deep_run_sorted = deep_run.sort_values("prob_SF", ascending=False).reset_index(drop=True)

display(deep_run_sorted.head(20))

# Stacked-ish bar chart for a subset (top 10–12 teams)
top_deep = deep_run_sorted.head(12)

x = np.arange(len(top_deep))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - width, top_deep["prob_QF"], width, label="Reach QF")
plt.bar(x,         top_deep["prob_SF"], width, label="Reach SF")
plt.bar(x + width, top_deep["prob_F"],  width, label="Reach Final")

plt.xticks(x, top_deep["team"], rotation=45, ha="right")
plt.ylabel("Probability")
plt.title("Deep run probabilities – Top 12 teams")
plt.legend()
plt.tight_layout()
plt.show()

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>group</th>
      <th>prob_QF</th>
      <th>prob_SF</th>
      <th>prob_F</th>
      <th>prob_W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>C</td>
      <td>0.1631</td>
      <td>0.1140</td>
      <td>0.0806</td>
      <td>0.1337</td>
    </tr>
    <tr>
      <th>1</th>
      <td>England</td>
      <td>L</td>
      <td>0.1651</td>
      <td>0.1126</td>
      <td>0.0670</td>
      <td>0.0997</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spain</td>
      <td>H</td>
      <td>0.1729</td>
      <td>0.1101</td>
      <td>0.0754</td>
      <td>0.1129</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Argentina</td>
      <td>J</td>
      <td>0.1553</td>
      <td>0.1089</td>
      <td>0.0729</td>
      <td>0.0953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>I</td>
      <td>0.1549</td>
      <td>0.1035</td>
      <td>0.0614</td>
      <td>0.0676</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Belgium</td>
      <td>G</td>
      <td>0.1602</td>
      <td>0.0975</td>
      <td>0.0532</td>
      <td>0.0499</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Portugal</td>
      <td>K</td>
      <td>0.1590</td>
      <td>0.0973</td>
      <td>0.0631</td>
      <td>0.0711</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Colombia</td>
      <td>K</td>
      <td>0.1566</td>
      <td>0.0906</td>
      <td>0.0496</td>
      <td>0.0530</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Netherlands</td>
      <td>F</td>
      <td>0.1454</td>
      <td>0.0799</td>
      <td>0.0432</td>
      <td>0.0429</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Italy</td>
      <td>B</td>
      <td>0.1405</td>
      <td>0.0773</td>
      <td>0.0375</td>
      <td>0.0279</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Germany</td>
      <td>E</td>
      <td>0.1464</td>
      <td>0.0756</td>
      <td>0.0344</td>
      <td>0.0257</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Denmark</td>
      <td>A</td>
      <td>0.1433</td>
      <td>0.0751</td>
      <td>0.0372</td>
      <td>0.0313</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Uruguay</td>
      <td>H</td>
      <td>0.1397</td>
      <td>0.0741</td>
      <td>0.0326</td>
      <td>0.0310</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Switzerland</td>
      <td>B</td>
      <td>0.1255</td>
      <td>0.0644</td>
      <td>0.0277</td>
      <td>0.0212</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morocco</td>
      <td>C</td>
      <td>0.1233</td>
      <td>0.0586</td>
      <td>0.0249</td>
      <td>0.0180</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Croatia</td>
      <td>L</td>
      <td>0.1234</td>
      <td>0.0549</td>
      <td>0.0238</td>
      <td>0.0140</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ecuador</td>
      <td>E</td>
      <td>0.1134</td>
      <td>0.0483</td>
      <td>0.0220</td>
      <td>0.0126</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Austria</td>
      <td>J</td>
      <td>0.0993</td>
      <td>0.0477</td>
      <td>0.0171</td>
      <td>0.0105</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Norway</td>
      <td>I</td>
      <td>0.0998</td>
      <td>0.0402</td>
      <td>0.0161</td>
      <td>0.0098</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Japan</td>
      <td>F</td>
      <td>0.0844</td>
      <td>0.0388</td>
      <td>0.0179</td>
      <td>0.0077</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](wc2026_analysis_files/wc2026_analysis_13_1.png)
    


## 5. Match-level examples – Upsets & balanced fixtures

The Poisson model also gives **match-level** probabilities for each group-stage game:

- Expected goals per team (`lambda_home`, `lambda_away`)
- Win/draw/loss probabilities (`p_home_win`, `p_draw`, `p_away_win`)

We can use this to:
- Identify potential **upsets**
- Highlight the most **balanced fixtures**



```python
# Copy with helper columns
m = group_match_probs.copy()
m["favoured_team"] = np.where(
    m["p_home_win"] > m["p_away_win"],
    m["home_team"],
    m["away_team"],
)
m["favoured_prob"] = m[["p_home_win", "p_away_win"]].max(axis=1)

# Potential upsets: favoured_prob not too high (say < 0.6)
possible_upsets = m.sort_values("favoured_prob").head(10)

print("Most balanced / potential upset matches:")
display(possible_upsets[[
    "date", "group", "home_team", "away_team",
    "lambda_home", "lambda_away",
    "p_home_win", "p_draw", "p_away_win",
    "favoured_team", "favoured_prob"
]])

```

    Most balanced / potential upset matches:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>group</th>
      <th>home_team</th>
      <th>away_team</th>
      <th>lambda_home</th>
      <th>lambda_away</th>
      <th>p_home_win</th>
      <th>p_draw</th>
      <th>p_away_win</th>
      <th>favoured_team</th>
      <th>favoured_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>2026-06-26</td>
      <td>H</td>
      <td>Cape Verde</td>
      <td>Saudi Arabia</td>
      <td>0.7937</td>
      <td>0.8099</td>
      <td>0.3190</td>
      <td>0.3528</td>
      <td>0.3282</td>
      <td>Saudi Arabia</td>
      <td>0.3282</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2026-06-26</td>
      <td>G</td>
      <td>Egypt</td>
      <td>Iran</td>
      <td>0.8025</td>
      <td>0.8754</td>
      <td>0.3081</td>
      <td>0.3429</td>
      <td>0.3490</td>
      <td>Iran</td>
      <td>0.3490</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2026-06-14</td>
      <td>F</td>
      <td>Poland</td>
      <td>Tunisia</td>
      <td>0.8831</td>
      <td>0.9610</td>
      <td>0.3171</td>
      <td>0.3237</td>
      <td>0.3592</td>
      <td>Tunisia</td>
      <td>0.3592</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2026-06-19</td>
      <td>D</td>
      <td>United States</td>
      <td>Australia</td>
      <td>0.9876</td>
      <td>1.0376</td>
      <td>0.3340</td>
      <td>0.3060</td>
      <td>0.3600</td>
      <td>Australia</td>
      <td>0.3600</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2026-06-22</td>
      <td>I</td>
      <td>Norway</td>
      <td>Senegal</td>
      <td>1.0703</td>
      <td>1.0071</td>
      <td>0.3656</td>
      <td>0.3013</td>
      <td>0.3331</td>
      <td>Norway</td>
      <td>0.3656</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2026-06-19</td>
      <td>D</td>
      <td>Turkey</td>
      <td>Paraguay</td>
      <td>1.1223</td>
      <td>1.0467</td>
      <td>0.3724</td>
      <td>0.2935</td>
      <td>0.3341</td>
      <td>Turkey</td>
      <td>0.3724</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2026-06-27</td>
      <td>K</td>
      <td>Colombia</td>
      <td>Portugal</td>
      <td>1.0282</td>
      <td>1.1102</td>
      <td>0.3311</td>
      <td>0.2960</td>
      <td>0.3729</td>
      <td>Portugal</td>
      <td>0.3729</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2026-06-18</td>
      <td>A</td>
      <td>Mexico</td>
      <td>South Korea</td>
      <td>1.1141</td>
      <td>0.9939</td>
      <td>0.3817</td>
      <td>0.2981</td>
      <td>0.3201</td>
      <td>Mexico</td>
      <td>0.3817</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2026-06-20</td>
      <td>F</td>
      <td>Tunisia</td>
      <td>Japan</td>
      <td>0.8541</td>
      <td>1.0118</td>
      <td>0.2974</td>
      <td>0.3203</td>
      <td>0.3823</td>
      <td>Japan</td>
      <td>0.3823</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2026-06-12</td>
      <td>D</td>
      <td>United States</td>
      <td>Paraguay</td>
      <td>1.0179</td>
      <td>0.8571</td>
      <td>0.3835</td>
      <td>0.3193</td>
      <td>0.2972</td>
      <td>United States</td>
      <td>0.3835</td>
    </tr>
  </tbody>
</table>
</div>


## 6. Limitations & Modelling Choices

Some important caveats and modelling choices to note (for your README):

- **Data window:** Only matches from 2018–2025 are used, which reflects recent form but may overweight short-term trends.
- **Team-level model:** The Poisson model uses team-level attack/defence parameters and a global home advantage. It does not model:
  - Injuries
  - Squad rotation
  - Individual player strengths
- **Poisson assumptions:**
  - Goals are modelled as independent Poisson random variables for home and away teams.
  - This ignores tactical “game state” effects (e.g. teams shutting down at 1–0).
- **Knockout structure:**
  - A simplified 32-team bracket is used with random seeding of qualified teams.
  - Official FIFA 2026 third-place rules and fixed bracket positions are not replicated exactly (by design, to keep the project understandable and reproducible).
- **No betting odds or Elo ratings:**
  - The model is entirely driven by fixture outcomes and goal counts in international matches.
  - You can mention that this is a “purely data-driven Poisson model” without market information.

Despite these limitations, the pipeline is:
- fully reproducible,
- conceptually transparent, and
- strong enough for a GitHub portfolio project.


## 7. Possible Extensions

If you want to extend the project further:

1. **More realistic knockout bracket**
   - Implement the official FIFA 2026 structure with:
     - Fixed positions for group winners/runners-up
     - Correct mapping of third-place teams

2. **Time-decayed weights**
   - Give more weight to recent matches (e.g. exponential decay based on match date).

3. **Home/continent effects**
   - Add extra advantage for host confederation teams (USA, Mexico, Canada or CONCACAF).

4. **Interactive dashboard**
   - Build a Streamlit or Plotly Dash app that lets users:
     - Adjust number of simulations
     - Inspect probabilities by team or group
     - Explore hypothetical changes (e.g. injuries, form shocks)

5. **Comparison with betting odds**
   - Collect bookmaker implied probabilities before the tournament and compare:
     - Which teams the model over/underestimates relative to the market.

