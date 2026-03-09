import numpy as np
import pandas as pd
from models.poisson_model import PoissonModel


class HybridPredictor:
    """
    Combines Random Forest classification with Poisson probabilities to output blended
    match outcome predictions.
    """

    def __init__(self, rf_model, league_avg: float, rf_weight: float = 0.5) -> None:
        """
        Constructs a new instance of HybridPredictor.

        Args:
            rf_model (_type_): Random Forest model.
            league_avg (float): League average goals per team per game, used to
                compute expected goals for the Poisson model.
            rf_weight (float, optional): Weight given to RF probabilities that
                determines how much importance the input data has on the model output.
                Defaults to 0.5.
        """
        self.rf_model = rf_model
        self.poisson = PoissonModel()
        self.league_avg = league_avg
        self.rf_weight = rf_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate blended match outcome predictions for a set of matches.

        Args:
            X (pd.DataFrame): Feature DataFrame for test matches. Must contain the columns
                used by the RF model (home_form_7, away_form_7, etc.) as well as home_att,
                away_def, home_def, away_att for Poisson xG calculation.

        Returns:
            np.ndarray: Array of predicted match outcomes ("H", "D", or "A") with the same
                length as the number of rows in X.
        """
        # RF probabilities - shape (n_matches, 3)
        rf_probs = self.rf_model.predict_proba(X)

        # Poisson probabilities
        poisson_probs = []
        for _, row in X.iterrows():
            home_xg = row["home_att"] * row["away_def"] * self.league_avg
            away_xg = row["away_att"] * row["home_def"] * self.league_avg

            probs = self.poisson.win_probabilities_by_goals(home_xg, away_xg)
            poisson_probs.append([probs["home_win"], probs["draw"], probs["away_win"]])

        poisson_probs = np.array(poisson_probs)

        # Blend
        blended = (self.rf_weight * rf_probs) + ((1 - self.rf_weight) * poisson_probs)

        # Pick the class with highest blended probability
        classes = self.rf_model.classes_
        predictions = classes[np.argmax(blended, axis=1)]

        return predictions
