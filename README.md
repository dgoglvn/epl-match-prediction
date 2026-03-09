# EPL Match Outcome Predictor

predicting the premier league matches for the 2025/26 season

A machine learning system that predicts English Premier League match outcomes (Home Win, Draw, Away Win) using a Random Forest classifier combined with a Poisson model.

Based on two research papers:

- Ulmer & Fernandez, _Predicting Soccer Match Results in the English Premier League_ (Stanford CS229, 2014)
- Harit & Mody, _Predicting English Premier League Winners using Machine Learning_ (UMass CS589, 2017)

## Results

| Model                             | Accuracy |
| --------------------------------- | -------- |
| Random baseline                   | ~0.33    |
| BBC analyst Mark Lawrenson        | 0.52     |
| Ulmer best (SGD)                  | 0.52     |
| Harit best (RF)                   | 0.55     |
| **This project — RF baseline**    | **0.64** |
| **This project — RF tuned**       | **0.65** |
| Hybrid (RF + Poisson, 0.7 weight) | 0.64     |

Best hyperparameters found via GridSearchCV: `criterion=entropy`, `max_features=sqrt`, `min_samples_split=7`, `n_estimators=150`.

### Per-Class Performance

| Class    | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| Home Win | 0.68      | 0.67   | 0.67 |
| Draw     | 0.33      | 0.13   | 0.19 |
| Away Win | 0.66      | 0.87   | 0.75 |

Draw prediction remains the weakest class, consistent with findings from both reference papers. The model performs strongest on away win prediction.

## How It Works

The system has three stages: data ingestion, feature engineering, and prediction.

**Data Ingestion** loads 25 seasons seasons of EPL match results (2000/01-2024/25) from football-data.co.uk CSVs, normalizes team names across seasons, and concatenates everything into a single DataFrame.

**Feature Engineering** computes the following features for each match using only pre-match data to prevent data leakage: `home_form_7`, `away_form_7`, `home_att`, `away_att`, `home_def`, `away_def`, `home_goal_diff`, `away_goal_diff`, `home_win_pct`, `away_win_pct`, `poisson_home_xg`, `poisson_away_xg`.

All cumulative features use `.expanding().mean().shift(1)` to ensure each matchday only sees data from prior games.

**Prediction** uses a tuned Random Forest classifier. A Hybrid Predictor blends RF probabilities with Poisson-derived win/draw/loss probabilities using a configurable weight parameter, though results show the RF is dominant at weights >= 0.7.

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dgoglvn/epl-match-prediction.git
cd epl-match-prediction
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download historical match data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php). Download the CSV for each EPL season (E0.csv) and save them in `data/historical/` with the naming format `YYYY-YY.csv` (e.g., `2000-01.csv`).

### Running

From the project root directory:

```bash
python3 main.py
```

This will load all historical seasons, build features, train the base and tuned Random Forest models, output accuracy scores, display the confusion matrix, and run the hybrid predictor.

## References

- Ulmer, B. & Fernandez, M. (2014). _Predicting Soccer Match Results in the English Premier League._ Stanford University, CS229.
- Harit, D. & Mody, R. (2017). _Predicting English Premier League Winners using Machine Learning._ Univeristy of Massachusetts Amherst, CS589.
- Football-Data.co.uk — [https://www.football-data.co.uk/englandm.php](https://www.football-data.co.uk/englandm.php)
