import pandas as pd

from models.poisson_model import PoissonModel
from data.stats_calculator import StatsCalculator
from core.team import Team

model = PoissonModel()
stats_calculator = StatsCalculator()
# team = Team("Liverpool")

teams = stats_calculator.compute_team_ratings()
pd.set_option('display.max_rows', None)
for i in range(0, 2):
    print(f"{teams["Team"][i]}, {teams["ATT"]}, {teams["DEF"][i]}")

# df = pd.DataFrame(stats_calculator.compute_team_ratings())
# pd.set_option('display.max_rows', None)
#
# filtered_df = df.loc[df["Team"] == "Arsenal"]
#
# print(filtered_df["DEF"].iloc[0])