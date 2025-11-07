from data.stats_calculator import StatsCalculator
from core.league import League
from core.team import Team

league: League = League()

stats_calculator: StatsCalculator = StatsCalculator()
team: Team = Team("Leeds United")
# print(stats_calculator.compute_team_ratings())
# print(league.get_team_stats("Liverpool"))
# print(league.get_team_ratings())

print(team.get_rating())