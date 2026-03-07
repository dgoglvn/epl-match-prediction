import pandas as pd
import numpy as np
import os


class HistoricalDataLoader:
    """
    Reads and concatenates multiple seasons CSVs into a single DataFrame.
    """

    def __init__(self, directory="data/historical/") -> None:
        self.directory = directory

        # Normalize team names so they are consistent
        self.team_name_map = {
            "Man United": "Manchester United",
            "Man City": "Manchester City",
            "Spurs": "Tottenham",
            "Nott'm Forest": "Nottingham Forest",
            "Sheffield Weds": "Sheffield Wednesday",
        }

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
        ]

        # These columns are are available 2001/02 onward
        self.MATCH_STAT_COLUMNS = [
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

    def normalize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles team name normalization (team names change or get relegated/promoted).
        Ensures team names are kept consistent (e.g., Spurs, Tottenham Hotspur -> Tottenham).

        Args:
            df (pd.DataFrame): DataFrame to go through and normalize names.

        Returns:
            pd.DataFrame: New DataFrame after normalizing names.
        """
        df["HomeTeam"] = df["HomeTeam"].replace(self.team_name_map)
        df["AwayTeam"] = df["AwayTeam"].replace(self.team_name_map)
        return df

    def load_single_season(self, year: int) -> pd.DataFrame:
        """
        Helper method for loading and cleaning a csv file.

        Args:
            year (int): Year the season was played.

        Returns:
            pd.DataFrame: The cleaned DataFrame so we can concatenate to the main csv.
        """
        filename = f"{year}-{str(year + 1)[-2:]}.csv"
        filepath = os.path.join(self.directory, filename)
        df = pd.read_csv(filepath, on_bad_lines="skip", encoding="latin-1")

        all_columns = self.CORE_COLUMNS + self.MATCH_STAT_COLUMNS

        # List comprehension trick for filtering through and keeping only the columns we need
        cols_to_keep = [col for col in all_columns if col in df.columns]
        df = df[cols_to_keep]

        # Add column that shows the season
        season = f"{year}-{str(year + 1)[-2:]}"
        df["Season"] = season

        self.normalize_team_names(df)

        # Drop NaN (Not a Number) values
        df = df.dropna()

        return df

    def load_all_seasons(self) -> pd.DataFrame:
        """
        Load and concatenate all CSV files.

        Returns:
            pd.DataFrame: DataFrame containing all PL seasons from 1993-2025.
        """
        all_dfs = []

        # for year in range(1993, 2026)
        for year in range(1993, 2025):
            df = self.load_single_season(year)
            all_dfs.append(df)

        all_seasons = pd.concat(all_dfs, ignore_index=True)

        # Format "Date" column
        if "Date" in all_seasons.columns:
            all_seasons["Date"] = pd.to_datetime(
                all_seasons["Date"], dayfirst=True, format="mixed"
            )

        all_seasons = all_seasons.sort_values("Date")

        return all_seasons


# Testing purposes
if __name__ == "__main__":
    loader = HistoricalDataLoader("data/historical/")
    df = loader.load_all_seasons()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())
    print(df.tail())
    print(df["HomeTeam"].unique())

    # Write to CSV
    # df.to_csv("data/historical/*.csv", sep=",")
