import pandas as pd

class DataLoader:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def load_league_table(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_name)

        return df