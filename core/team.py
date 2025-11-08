# team.py
import pandas as pd
from core.league import League

class Team:
    def __init__(self, team_name: str) -> None:
        self.team_name = team_name
        self.league: League = League()

    def get_team_name(self) -> str:
        return self.team_name

    def get_rating(self) -> str:
        team_ratings_df: pd.DataFrame = self.league.get_team_ratings()
        row = team_ratings_df[team_ratings_df["Team"].str.lower() == self.team_name.lower()]
        if row.empty:
            raise ValueError(f"Team name '{self.team_name}' not found.")

        return f"{self.team_name}: ATT={row.iloc[0]['ATT']}, DEF={row.iloc[0]['DEF']}"