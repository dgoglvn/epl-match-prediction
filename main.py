# main.py
from data.stats_calculator import StatsCalculator
from core.league import League
from core.team import Team
from models.poisson_model import PoissonModel

def main() -> None:
    league: League = League()
    avg_goals: float = league.get_avg_goals()
    print(f"League average goals per team per match: {avg_goals}")

    stats: StatsCalculator = StatsCalculator()

    home_team: Team = Team("Sunderland")
    away_team: Team = Team("Arsenal")

    # expected goals based on ATT/DEF and league averages
    home_xg, away_xg = stats.goal_expectancy(home_team.get_team_name(), away_team.get_team_name())

    print("\nExpected goals (xG):")
    print(f"{home_team.get_team_name()}: {home_xg}")
    print(f"{away_team.get_team_name()}: {away_xg}")

    # use Poisson model to get win/draw/loss probabilities
    model: PoissonModel = PoissonModel()
    probs = model.win_probabilities_by_goals(home_xg, away_xg)

    print("\nOverall match outcome probabilities:")
    print(f"{home_team.get_team_name()} win: {round(probs['home_win'], 2) * 100}%")
    print(f"Draw: {round(probs['draw'], 2) * 100}%")
    print(f"{away_team.get_team_name()} win: {round(probs['away_win'], 2) * 100}%")

    print("\nBreakdown - home team wins by exact goals scored:")
    for goals, p in probs["home_by_goals"].items():
        if p > 0:
            print(f"{home_team.get_team_name()} scores {goals}: {p:.3%} chance of winning")

    print("\nBreakdown - away team wins by exact goals scored:")
    for goals, p in probs["away_by_goals"].items():
        if p > 0:
            print(f"{away_team.get_team_name()} scores {goals}: {p:.3%} chance of winning")

if __name__ == "__main__":
    main()