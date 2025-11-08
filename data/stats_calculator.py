# stats_calculator.py
import pandas as pd
from data.data_loader import DataLoader
class StatsCalculator:
    def __init__(self) -> None:
        self.file_name: str = "data/Premier League Matchweek 11 Standings 25-26.csv"
        self.data_loader: DataLoader = DataLoader(self.file_name)
        self.df: pd.DataFrame = self.data_loader.load_league_table()

    def compute_league_avg_goals(self) -> float:
        gf_sum = 0
        for value in self.df.loc[:, "GF"]:
            gf_sum += value

        matches_played_sum: int = 0
        for value in self.df.loc[:, "MP"]:
            matches_played_sum += value

        return gf_sum / matches_played_sum

    def compute_att_rating(self, team: str) -> float:
        league_avg_goals: float = self.compute_league_avg_goals()

        # get team's GF stat
        teams_gf: int = self.df.loc[self.df["Team"] == team, "GF"].iloc[0]

        # get team's number of matches played
        matches_played: int = self.df.loc[self.df["Team"] == team, "MP"].iloc[0]

        return round((teams_gf / matches_played) * (1 / league_avg_goals), 3)

    def compute_def_rating(self, team: str) -> float:
        league_avg_goals: float = self.compute_league_avg_goals()

        # get team's GA stat
        teams_ga: int = self.df.loc[self.df["Team"] == team, "GA"].iloc[0]

        # get team's number of matches played
        matches_played: int = self.df.loc[self.df["Team"] == team, "MP"].iloc[0]

        return round((teams_ga / matches_played) * (1 / league_avg_goals), 3)

    def compute_team_ratings(self) -> pd.DataFrame:
        teams: list[str] = self.df["Team"].unique().tolist()
        results: list[dict[str, float | str]] = []

        for team in teams:
            att_rating: float = self.compute_att_rating(team)
            def_rating: float = self.compute_def_rating(team)

            results.append({
                "Team": team,
                "ATT": att_rating,
                "DEF": def_rating,
            })

        return pd.DataFrame(results)

    def goal_expectancy(self, home_team_name: str, away_team_name: str) -> tuple[float, float]:
        home_team_att: float = self.compute_att_rating(home_team_name)
        away_team_def: float = self.compute_def_rating(away_team_name)
        league_avg_goals: float = self.compute_league_avg_goals()

        away_team_att: float = self.compute_att_rating(away_team_name)
        home_team_def: float = self.compute_def_rating(home_team_name)

        home_team_xg: float = home_team_att * away_team_def * league_avg_goals
        away_team_xg: float = away_team_att * home_team_def * league_avg_goals

        return round(home_team_xg, 2), round(away_team_xg, 2)