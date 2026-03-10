import numpy as np
import pandas as pd


class FormCalculator:
    """
    Computes form for each team in a match based on their previous 7 matches.
    """

    def compute_team_form(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        For each match, look at the previous 7 matches to compute form.
        +3 for win, +1 for draw, +0 for loss.

        Args:
            team_name (str): Team name.

        Returns:
            pd.DataFrame: DataFrame with new columns: "P" (points) and "form_7"
            (computed form from past 7 games).
        """
        # Get all matches played by the specified team
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
        team_matches["P"] = np.nan

        for idx, row in team_matches.iterrows():
            if (row.FTR == "H" and row.HomeTeam == team) or (
                row.FTR == "A" and row.AwayTeam == team
            ):
                team_matches.at[idx, "P"] = 3
            elif row.FTR == "D":
                team_matches.at[idx, "P"] = 1
            else:
                team_matches.at[idx, "P"] = 0

        # For each match, look at the previous 7 games to compute form
        team_matches["form_7"] = (
            team_matches.groupby("Season")[
                "P"
            ]  # Groups by season so form doesn't carry over to the next
            .apply(lambda x: x.rolling(7).sum())
            .reset_index(level=0, drop=True)
        )

        team_matches = team_matches.drop(columns=["P"])
        return team_matches

    def add_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add home_form_7 and away_form_7 to the full match DataFrame.

        Args:
            df (pd.DataFrame): Full match DataFrame.

        Returns:
            pd.DataFrame: Same DataFrame with two new columns: home_form_7, away_form_7.
        """
        df["home_form_7"] = np.nan
        df["away_form_7"] = np.nan

        # Computing team form for each team for each match
        for team in df["HomeTeam"].unique():
            team_form = self.compute_team_form(df, team)

            home_mask = team_form["HomeTeam"] == team
            df.loc[team_form.index[home_mask], "home_form_7"] = team_form.loc[
                home_mask, "form_7"
            ].values  # type: ignore

            away_mask = team_form["AwayTeam"] == team
            df.loc[team_form.index[away_mask], "away_form_7"] = team_form.loc[
                away_mask, "form_7"
            ].values  # type: ignore

        return df
