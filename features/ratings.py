import numpy as np
import pandas as pd


class RatingsCalculator:
    """
    Computes cumulative ATT and DEF ratings for each team at each matchday.

    ATT rating measures a team's scoring ability relative to the league average.
    DEF rating measures how many goals a team concedes relative to the league average.

    ATT > 1.0 means the team scores more than average.
    DEF < 1.0 means the team concedes less than average (good defense).

    All ratings are computed using only data available before each match
    to prevent data leakage.

    These ratings serve as:
        1. Direct features for the Random Forest classifier.
        2. Inputs to the Poisson model for computing expected goals (xG),
            which become additional meta-features for the classifier.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Args:
            df (pd.DataFrame): Historical match DataFrame with columns including
            HomeTeam, AwayTeam, FTHG (Full Time Home Goals),
            and FTAG (Full Time Away Goals).
        """
        self.df = df

    def calculate_league_avg_goals(self, df: pd.DataFrame) -> float:
        """
        Compute the average goals scored per team per game across the league.

        Formula: (total home goals + total away goals) / (number of matches * 2)

        We divide by 2 because each match invovles two teams, and we want
        the per-team average, not the per-match total.

        Args:
            df (pd.DataFrame): Match DataFrame with FTHG and FTAG columns.

        Returns:
            float: Average goals per team per game.
        """
        total_goals = df["FTHG"].sum() + df["FTAG"].sum()
        return total_goals / (len(df) * 2)

    def compute_team_ratings(self, team: str) -> pd.DataFrame:
        """
        Compute cumulative ATT and DEF ratings for a single team across all
        their matches.

        Steps:
            1. Filter to all matches involving the team.
            2. Create GF/GA columns from the team's perspective using np.where
                (since FTHG/FTAG are from the home/away perspective).
            3. Compute expanding mean of GF and GA to get cumulutive goals
                per game, shifted by 1 to exclude the current match.
            4. Divide by league average to get ATT and DEF ratings.

        Args:
            team (str): Team name.

        Returns:
            pd.DataFrame: DataFrame of the team's matches with added columns:
            GF, GA, gf_per_game, ga_per_game, ATT, DEF.
            First row will have NaN ratings (no prior data available).
        """
        # Get all matches played by the team
        team_matches = self.df[
            (self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team)
        ].copy()

        # Creating a new column "GF". If the team is the home team, use FTHG value for GF.
        # Otherwise (team is the away team), use FTAG value for GF.
        team_matches["GF"] = np.where(
            team_matches["HomeTeam"] == team, team_matches["FTHG"], team_matches["FTAG"]
        )
        # Creating a new column "GA". If the team is the home team, use FTAG value for GA.
        # Otherwise (team is the away team), use FTHG value for GA.
        team_matches["GA"] = np.where(
            team_matches["HomeTeam"] == team, team_matches["FTAG"], team_matches["FTHG"]
        )

        team_matches["gf_per_game"] = (
            team_matches.groupby("Season")["GF"]
            .apply(lambda x: x.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        team_matches["ga_per_game"] = (
            team_matches.groupby("Season")["GA"]
            .apply(lambda x: x.expanding().mean().shift(1))
            .reset_index(level=0, drop=True)
        )

        league_avg = self.calculate_league_avg_goals(self.df)
        team_matches["ATT"] = team_matches["gf_per_game"] / league_avg
        team_matches["DEF"] = team_matches["ga_per_game"] / league_avg

        return team_matches

    def add_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add home_att, home_def, away_att, and away_def columns to the
        full match DataFrame.

        Loops through every team, computes their ratings, then merges
        the values into the correct column based on whether the team
        was home or away in each match.

        Args:
            df (pd.DataFrame): Full match DataFrame from HistoricalDataLoader.

        Returns:
            pd.DataFrame: Same DataFrame with four new columns:
            home_att, home_def, away_att, away_def.
            Early-season rows will have NaN where ratings couldn't be computed
            (dropped later by FeatureEngineer.build_features).
        """

        df = df.copy()
        df["home_att"] = np.nan
        df["home_def"] = np.nan
        df["away_att"] = np.nan
        df["away_def"] = np.nan

        for team in df["HomeTeam"].unique():
            team_ratings = self.compute_team_ratings(team)

            home_mask = team_ratings["HomeTeam"] == team
            df.loc[team_ratings.index[home_mask], "home_att"] = team_ratings.loc[
                home_mask, "ATT"
            ].values  # type: ignore
            df.loc[team_ratings.index[home_mask], "home_def"] = team_ratings.loc[
                home_mask, "DEF"
            ].values  # type: ignore

            away_mask = team_ratings["AwayTeam"] == team
            df.loc[team_ratings.index[away_mask], "away_att"] = team_ratings.loc[
                away_mask, "ATT"
            ].values  # type: ignore
            df.loc[team_ratings.index[away_mask], "away_def"] = team_ratings.loc[
                away_mask, "DEF"
            ].values  # type: ignore

        return df
