import math

class PoissonModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals

    @staticmethod
    def poisson_pmf(self, x: int, lam: float) -> float:
        return (math.exp(-lam) * (lam**x)) / math.factorial(x)

    @staticmethod
    def to_be_named(self):
        pass

    def predict_match(self, home_team: str, away_team: str) -> dict:
        pass
