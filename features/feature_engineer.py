import pandas as pd
import numpy as np
import os


class FeatureEngineer:
    """
    Takes the historical DataFrame and produces one row per match with the several features.
    """

    def __init__(self, filepath="data/historical/2024-25.csv") -> None:
        self.filepath = filepath

        # The columns to keep
        self.CORE_COLUMNS = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "HTHG",
            "HTAG",
            "HTR",
            "HS",  # shots
            "AS",
            "HST",  # shots on target
            "AST",
            "HF",  # fouls
            "AF",
            "HC",  # corners
            "AC",
            "HY",  # yellow cards
            "AY",
            "HR",  # red cards
            "AR",
        ]

    def form_7(self, team_name: str) -> pd.DataFrame:
        """
        For each match, look at the previous 7 matches to compute form.
        +3 for win, +1 for draw, +0 for loss.

        Args:
            team_name (str): Team name.

        Returns:
            pd.DataFrame: DataFrame with new columns: "P" (points) and "form_7" (computed form from past 7 games).
        """
        df = pd.read_csv(self.filepath)

        # Handle team name input errors
        if team_name not in df["HomeTeam"].unique():
            print("Team name error")
            return pd.DataFrame()

        # Get all matches played by specified team, sorted by date
        team_matches = df[(df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)]

        cols_to_keep = [col for col in self.CORE_COLUMNS if col in team_matches.columns]
        team_matches = team_matches[cols_to_keep]

        team_matches["P"] = np.nan

        for idx, row in team_matches.iterrows():
            if (row.FTR == "H" and row.HomeTeam == team_name) or (
                row.FTR == "A" and row.AwayTeam == team_name
            ):
                team_matches.at[idx, "P"] = 3
            elif row.FTR == "D":
                team_matches.at[idx, "P"] = 1
            else:
                team_matches.at[idx, "P"] = 0

        # For each match, look at the previous 7 games to compute form
        team_matches["form_7"] = team_matches["P"].rolling(7).sum()
        team_matches["P"] = team_matches["P"].astype("int64")

        return team_matches


if __name__ == "__main__":
    df = FeatureEngineer("data/historical/2024-25.csv")
    print(df.form_7("Liverpool"))
