# poisson_model.py
import math
from typing import Dict, List, TypedDict

class WinProbabilities(TypedDict):
    home_by_goals: Dict[int, float]
    away_by_goals: Dict[int, float]
    home_win: float
    draw: float
    away_win: float

class PoissonModel:
    def __init__(self, max_goals: int = 10) -> None:
        self.max_goals = max_goals

    def _goal_distribution(self, lam: float) -> List[float]:
        # return P(X = k) for k = 0..max_goals for a Poisson(lambda) random variable
        return [self.poisson_pmf(k, lam) for k in range(self.max_goals + 1)]


    def win_probabilities_by_goals(
            self,
            home_team_xg: float,
            away_team_xg: float,
    ) -> WinProbabilities:
        # distributions P(X = k) and P(Y = k) for k = 0..max_goals
        home_goal_probs: List[float] = self._goal_distribution(home_team_xg)
        away_goal_probs: List[float] = self._goal_distribution(away_team_xg)

        home_by_goals: Dict[int, float] = {}
        away_by_goals: Dict[int, float] = {}

        # home: P(win with exactly x goals) for x = 0.._max_goals
        for x in range(self.max_goals + 1):
            p_home_x: float = home_goal_probs[x]
            # away can score 0..x-1
            p_away_less_than_x: float = sum(away_goal_probs[0:x])
            home_by_goals[x] = p_home_x * p_away_less_than_x

        # away: P(win with exactly y goals) for y = 0..max_goals
        for y in range(self.max_goals + 1):
            p_away_y: float = away_goal_probs[y]
            # home can score 0..y-1
            p_home_less_than_y: float = sum(home_goal_probs[0:y])
            away_by_goals[y] = p_away_y * p_home_less_than_y

        total_home_win: float = sum(home_by_goals.values())
        total_away_win: float = sum(away_by_goals.values())
        total_draw: float = sum(home_goal_probs[g] * away_goal_probs[g] for g in range(self.max_goals + 1))

        # tiny numerical/truncation errors can make the sum slightly != 1
        total_mass: float = total_home_win + total_away_win + total_draw
        home_win: float = total_home_win / total_mass
        draw: float = total_draw / total_mass
        away_win: float = total_away_win / total_mass

        return {
            "home_by_goals": home_by_goals,
            "away_by_goals": away_by_goals,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }

    @staticmethod
    def poisson_pmf(x: int, lam: float) -> float:
        return (math.exp(-lam) * (lam**x)) / math.factorial(x)