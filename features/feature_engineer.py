import pandas as pd
import numpy as np
import os


class FeatureEngineer:
    """
    Takes the historical DataFrame and produces one row per match with the several features.
    """

    def compute_team_form(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        For each match, look at the previous 7 matches to compute form.
        +3 for win, +1 for draw, +0 for loss.

        Args:
            team_name (str): Team name.

        Returns:
            pd.DataFrame: DataFrame with new columns: "P" (points) and "form_7" (computed form from past 7 games).
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
        team_matches["form_7"] = team_matches["P"].rolling(7).sum()
        # team_matches["P"] = team_matches["P"].astype("int64")
        team_matches = team_matches.drop(columns=["P"])

        return team_matches

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes in raw match data and returns the same DataFrame but with new
        feature columns attached. Computes form for every team.

        Args:
            df (pd.DataFrame): Raw DataFrame from load_all_seasons().

        Returns:
            pd.DataFrame: DataFrame with new features.
        """
        df["home_form_7"] = np.nan
        df["away_form_7"] = np.nan

        for team in df["HomeTeam"].unique():
            # Get this team's form for every match they played
            team_form = self.compute_team_form(df, team)

            # Find matches where this team was HOME and fill in their form
            home_mask = df["HomeTeam"] == team
            df.loc[home_mask, "home_form_7"] = team_form.loc[home_mask, "form_7"]

            # Find matches where this team was AWAY and fill in their form
            away_mask = df["AwayTeam"] == team
            df.loc[away_mask, "away_form_7"] = team_form.loc[away_mask, "form_7"]

        return df


# Testing purposes
if __name__ == "__main__":
    engineer = FeatureEngineer()
    raw_df = pd.read_csv("data/historical/2024-25.csv")
    featured_df = engineer.build_features(raw_df)
    print(
        featured_df[
            ["Date", "HomeTeam", "AwayTeam", "FTR", "home_form_7", "away_form_7"]
        ].to_string()
    )
