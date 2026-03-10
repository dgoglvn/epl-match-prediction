from typing import Any

import pandas as pd


class MatchPredictor:
    """
    Predicts upcoming fixtures using a trained Random Forest model and
    the current season's feature data.
    """

    FEATURE_COLUMNS = [
        "home_form_7",
        "away_form_7",
        "home_att",
        "home_def",
        "away_att",
        "away_def",
        "home_goal_diff",
        "away_goal_diff",
        "home_win_pct",
        "away_win_pct",
        "poisson_home_xg",
        "poisson_away_xg",
    ]

    def __init__(
        self, model, featured_df: pd.DataFrame, season: str = "2025-26"
    ) -> None:
        self.model = model
        self.season_df = featured_df[featured_df["Season"] == "2025-26"]

    def predict_match(self, home_team: str, away_team: str) -> dict[str, Any]:
        """
        Predict the outcome of a match between two teams.

        Uses each team's most recent feature values from the current season
        to build a featur vector, then runs it through the trained model.

        Args:
            home_team (str): Name of the home team (must match the DataFrame exactly).
            away_team (str): Name of the away team (must match the DataFrame exactly).

        Returns:
            dict[str, Any]: Dictionary with predicted class and probabilities.
        """
        # Get each team's most recent features
        home_matches = self.season_df[self.season_df["HomeTeam"] == home_team]
        away_matches = self.season_df[self.season_df["AwayTeam"] == away_team]

        home_row = home_matches.iloc[-1]
        away_row = away_matches.iloc[-1]

        match_features = {
            "home_form_7": home_row["home_form_7"],
            "away_form_7": away_row["away_form_7"],
            "home_att": home_row["home_att"],
            "home_def": home_row["home_def"],
            "away_att": away_row["away_att"],
            "away_def": away_row["away_def"],
            "home_goal_diff": home_row["home_goal_diff"],
            "away_goal_diff": away_row["away_goal_diff"],
            "home_win_pct": home_row["home_win_pct"],
            "away_win_pct": away_row["away_win_pct"],
            "poisson_home_xg": home_row["poisson_home_xg"],
            "poisson_away_xg": away_row["poisson_away_xg"],
        }

        X = pd.DataFrame([match_features])[self.FEATURE_COLUMNS]

        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        best_class = classes[probs.argmax()]

        result = {"prediction": best_class}
        for cls, prob in zip(classes, probs):
            result[label_map[cls]] = round(prob, 3)

        # Print for convenience
        print(f"\n{home_team} vs {away_team}")
        print(f"  Prediction: {label_map[best_class]}")
        for cls, prob in zip(classes, probs):
            print(f"  {label_map[cls]}: {prob:.1%}")

        return result
