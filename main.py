from models.random_forest_model import RandomForestModel


def main() -> None:
    """
    Entry point for the script that initializes the RandomForestModel.
    """
    model = RandomForestModel()
    model.run()


if __name__ == "__main__":
    main()
