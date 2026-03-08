import pandas as pd
import numpy as np
from features.form import FormCalculator
from features.ratings import RatingsCalculator
from features.goal_diff import GoalDiffCalculator
from features.win_pct import WinPctCalculator


class FeatureEngineer:
    """
    The orchestrator the calls helper classes.
    """

    def __init__(self) -> None:
        self.form = FormCalculator()

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
        ratings = RatingsCalculator(df)
        df = ratings.add_ratings(df)

        # Goal differential
        goal_diff = GoalDiffCalculator()
        df = goal_diff.add_goal_diff(df)

        # Win percentage
        win_pct = WinPctCalculator()
        df = win_pct.add_win_pct(df)

        # Poisson xG (depends on ratings being computed first)
        league_avg = ratings.calculate_league_avg_goals(df)
        df["poisson_home_xg"] = df["home_att"] * df["away_def"] * league_avg
        df["poisson_away_xg"] = df["away_att"] * df["home_def"] * league_avg

        # Drop rows where features couldn't be computed
        df = df.dropna()
        return df


if __name__ == "__main__":
    from data.historical_data_loader import HistoricalDataLoader

    loader = HistoricalDataLoader("data/historical/")
    raw_df = loader.load_all_seasons()

    engineer = FeatureEngineer()
    featured_df = engineer.build_features(raw_df)

    print(f"Shape: {featured_df.shape}")
    print(f"Columns: {featured_df.columns}")
    # print(
    #     featured_df[
    #         [
    #             "Date",
    #             "HomeTeam",
    #             "AwayTeam",
    #             "FTR",
    #             "home_form_7",
    #             "away_form_7",
    #             "home_att",
    #             "home_def",
    #             "away_att",
    #             "away_def",
    #             "home_goal_diff",
    #             "away_goal_diff",
    #             "home_win_pct",
    #             "away_win_pct",
    #             "poisson_home_xg",
    #             "poisson_away_xg",
    #         ]
    #     ].to_string()
    # )
