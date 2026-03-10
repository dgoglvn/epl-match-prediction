from sklearn.ensemble import RandomForestClassifier

from data.historical_data_loader import HistoricalDataLoader
from features.feature_engineer import FeatureEngineer
from models.match_predictor import MatchPredictor


def main() -> None:
    """
    Entry point for the script that initializes the RandomForestModel.
    """

    FEATURE_COLUMNS = [
        "home_form_7",
        "away_form_7",
        "home_att",
        "home_def",
        "away_att",
        "away_def",
        "home_goal_diff",
        "away_goal_diff",
        "home_win_pct",
        "away_win_pct",
        "poisson_home_xg",
        "poisson_away_xg",
    ]

    # Load the data
    loader = HistoricalDataLoader("data/historical/")
    raw_df = loader.load_all_seasons()

    engineer = FeatureEngineer()
    featured_df = engineer.build_features(raw_df)

    # Train on all data through 2024-25
    train_df = featured_df[featured_df["Season"] <= "2024-25"]
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FTR"]

    rf_model = RandomForestClassifier(
        criterion="entropy",
        max_features="sqrt",
        min_samples_split=7,
        n_estimators=150,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    predictor = MatchPredictor(rf_model, featured_df, season="2025-26")
    predictor.predict_match("Sunderland", "Brighton")

    # model = RandomForestModel()
    # model.run()


if __name__ == "__main__":
    main()
