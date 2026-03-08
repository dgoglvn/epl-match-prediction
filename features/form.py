import pandas as pd
import numpy as np


class FormCalculator:
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
