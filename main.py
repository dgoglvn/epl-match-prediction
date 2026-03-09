from models.random_forest_model import RandomForestModel
from models.hybrid_predictor import HybridPredictor


def main() -> None:
    """
    Entry point for the script that initializes the RandomForestModel.
    """
    model = RandomForestModel()
    model.run()


if __name__ == "__main__":
    main()
