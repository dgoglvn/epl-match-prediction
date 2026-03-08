from sklearn.ensemble import RandomForestClassifier
from data.historical_data_loader import HistoricalDataLoader
from features.feature_engineer import FeatureEngineer


def main() -> None:
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

    # Load data
    loader = HistoricalDataLoader("data/historical/")
    raw_df = loader.load_all_seasons()

    # Build features
    engineer = FeatureEngineer()
    featured_df = engineer.build_features(raw_df)

    # Chronological train/test split
    train_df = featured_df[featured_df["Season"] < "2023-24"]
    test_df = featured_df[featured_df["Season"] >= "2023-24"]

    # Prepare X and y
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FTR"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["FTR"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train seasons: {train_df['Season'].unique()}")
    print(f"Test seasons: {test_df['Season'].unique()}")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train (fit) the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = rf_model.predict(X_test)

    # Calculate accuracy
    accuracy = rf_model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
