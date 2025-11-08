# league.py
import pandas as pd
from data.data_loader import DataLoader
from data.stats_calculator import StatsCalculator

class League:
    file_name: str = "data/Premier League Matchweek 11 Standings 25-26.csv"
    data_loader: DataLoader = DataLoader(file_name)
    df: pd.DataFrame = data_loader.load_league_table()

    stats_calculator: StatsCalculator = StatsCalculator()

    def get_teams(self):
        return self.df.Team.unique()

    def get_league_standings(self) -> pd.DataFrame:
        return self.df

    def get_team_stats(self, team_name: str):
        row = self.df[self.df["Team"].str.lower() == team_name.lower()]
        if row.empty:
            raise ValueError(f"Team '{team_name}' not found.")
        return row.iloc[0]

    def get_team_ratings(self) -> pd.DataFrame:
        return self.stats_calculator.compute_team_ratings()

    def get_avg_goals(self) -> float:
        return self.stats_calculator.compute_league_avg_goals()