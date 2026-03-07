import pandas as pd
import numpy as np
from features.form import Form_Calculator
from features.ratings import Ratings_Calculator


class FeatureEngineer:
    """
    The orchestrator the calls helper classes.
    """

    def __init__(self) -> None:
        self.form = Form_Calculator()

    def add_form(self, df: pd.DataFrame) -> pd.DataFrame:
        df["home_form_7"] = np.nan
        df["away_form_7"] = np.nan

        # Computing team form for each team for each match
        for team in df["HomeTeam"].unique():
            team_form = self.form.compute_team_form(df, team)

            home_mask = team_form["HomeTeam"] == team
            df.loc[team_form.index[home_mask], "home_form_7"] = team_form.loc[home_mask, "form_7"].values  # type: ignore

            away_mask = team_form["AwayTeam"] == team
            df.loc[team_form.index[away_mask], "away_form_7"] = team_form.loc[away_mask, "form_7"].values  # type: ignore

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Form
        df = self.add_form(df)

        # ATT/DEF ratings
        ratings = Ratings_Calculator(df)
        df = ratings.add_ratings(df)

        # Poisson xG (depends on ratings being computed first)
        league_avg = ratings.calculate_league_avg_goals(df)
        df["poisson_home_xg"] = df["home_att"] * df["away_def"] * league_avg
        df["poisson_away_xg"] = df["away_att"] * df["home_def"] * league_avg

        # Drop rows where features couldn't be computed
        df = df.dropna()

        return df
