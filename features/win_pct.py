import pandas as pd
import numpy as np


class WinPctCalculator:
    """
    Computes win percentages for each team (home and away) per match.
    """
    def compute_win_pct(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        Computes win percentage for a single team.

        Args:
            df (pd.DataFrame): Full match DataFrame.
            team (str): Team name.

        Returns:
            pd.DataFrame: DataFrame of the team's matches with an added WPct column.
            First row will be NaN (no prior data).
        """
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()

        # If the team is the home team, their goals are FTHG. Otherwise, FTAG.
        # Gonna need the team's number of wins up until each match day, and
        # total number of games played.
        for idx, row in team_matches.iterrows():
            if (row.FTR == "H" and row.HomeTeam == team) or (
                row.FTR == "A" and row.AwayTeam == team
            ):
                team_matches.at[idx, "wins"] = 1
            else:
                team_matches.at[idx, "wins"] = 0

        team_matches["WPct"] = team_matches["wins"].expanding().mean().shift(1)
        team_matches["WPct"] = (
            team_matches.groupby("Season")["wins"]
            .apply(lambda x: x.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )

        return team_matches

    def add_win_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        df["home_win_pct"] = np.nan
        df["away_win_pct"] = np.nan

        for team in df["HomeTeam"].unique():
            team_wpct = self.compute_win_pct(df, team)

            home_mask = team_wpct["HomeTeam"] == team
            df.loc[team_wpct.index[home_mask], "home_win_pct"] = team_wpct.loc[home_mask, "WPct"].values  # type: ignore

            away_mask = team_wpct["AwayTeam"] == team
            df.loc[team_wpct.index[away_mask], "away_win_pct"] = team_wpct.loc[away_mask, "WPct"].values  # type: ignore

        return df
