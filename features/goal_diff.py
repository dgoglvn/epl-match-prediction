import pandas as pd
import numpy as np


class GoalDiffCalculator:
    """
    Computes cumulative goal differential for each team at each matchday.

    Goal differential is the running total of (goals scored - goals conceded)
    across all previous matches. A positive value means the team has scored
    more than they've conceded so far in the season.

    Uses .expanding().sum().shift(1) to ensure each match only see data
    from prior games.
    """

    def compute_team_goal_diff(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        Compute cumulative goal differential for a single team.

        Args:
            df (pd.DataFrame): Full match DataFrame.
            team (str): Team name.

        Returns:
            pd.DataFrame: DataFrame of the team's matches with an added GD column.
            First row will be NaN (no prior data).
        """
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()

        team_matches["GF"] = np.where(
            team_matches["HomeTeam"] == team, team_matches["FTHG"], team_matches["FTAG"]
        )
        team_matches["GA"] = np.where(
            team_matches["HomeTeam"] == team, team_matches["FTAG"], team_matches["FTHG"]
        )

        team_matches["GF_minus_GA"] = team_matches["GF"] - team_matches["GA"]

        team_matches["GD"] = (
            team_matches.groupby("Season")["GF_minus_GA"]
            .apply(lambda x: x.expanding().sum().shift(1))
            .reset_index(level=0, drop=True)
        )

        return team_matches

    def add_goal_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add home_goal_diff and away_goal_diff columns to the full match DataFrame.

        Args:
            df (pd.DataFrame): Full match DataFrame.

        Returns:
            pd.DataFrame: Same DataFrame with two new columns: home_goal_diff, away_goal_diff.
        """
        df["home_goal_diff"] = np.nan
        df["away_goal_diff"] = np.nan

        for team in df["HomeTeam"].unique():
            team_gd = self.compute_team_goal_diff(df, team)

            home_mask = team_gd["HomeTeam"] == team
            df.loc[team_gd.index[home_mask], "home_goal_diff"] = team_gd.loc[home_mask, "GD"].values  # type: ignore

            away_mask = team_gd["AwayTeam"] == team
            df.loc[team_gd.index[away_mask], "away_goal_diff"] = team_gd.loc[away_mask, "GD"].values  # type: ignore

        return df
