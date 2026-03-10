import pandas as pd

from features.form import FormCalculator
from features.goal_diff import GoalDiffCalculator
from features.ratings import RatingsCalculator
from features.win_pct import WinPctCalculator


class FeatureEngineer:
    """
    The orchestrator the calls all helper classes:
    FormCalculator, RatingsCalculator, GoalDiffCalculator, WinPctCalculator.

    Poisson model
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Manages the calculations of all feature sets and merges them into the
        main DataFrame.

        This method serially applies form, team ratings, goal differentials,
        win percentages. It also computes expected goals (xG) using a Poisson
        distribution based on the generated team ratings.

        Args:
            df (pd.DataFrame): The raw historical match data.

        Returns:
            pd.DataFrame: A processed DataFrame containing all engineered
            features, with rows containing NaNs removed.
        """
        # Form
        form = FormCalculator()
        df = form.add_form(df)

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
