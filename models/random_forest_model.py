from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from data.historical_data_loader import HistoricalDataLoader
from features.feature_engineer import FeatureEngineer
from models.hybrid_predictor import HybridPredictor
from features.ratings import RatingsCalculator


class RandomForestModel:
    """
    Uses the Random Forest Classifier from scikit-learn. Train, test, and evaluate
    the base Random Forest model and the tuned model with optimized hyperparameters.
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

    def run(self) -> None:
        # Load data
        loader = HistoricalDataLoader("data/historical/")
        raw_df = loader.load_all_seasons()

        ratings = RatingsCalculator(raw_df)
        league_avg = ratings.calculate_league_avg_goals(raw_df)

        # Build features
        engineer = FeatureEngineer()
        featured_df = engineer.build_features(raw_df)

        # Chronological train/test split
        train_df = featured_df[featured_df["Season"] < "2023-24"]
        test_df = featured_df[featured_df["Season"] >= "2023-24"]

        # Prepare X and y
        X_train = train_df[self.FEATURE_COLUMNS]
        y_train = train_df["FTR"]

        X_test = test_df[self.FEATURE_COLUMNS]
        y_test = test_df["FTR"]

        # print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        # print(f"Train seasons: {train_df['Season'].unique()}")
        # print(f"Test seasons: {test_df['Season'].unique()}")
        # print(f"\n{X_test[["home_att", "away_def", "home_def", "away_att"]]}")

        # ========== BASE MODEL ==========
        print("\n========== BASE MODEL ==========")

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train (fit) the model on the training data
        rf_model.fit(X_train, y_train)

        rf_probs = rf_model.predict_proba(X_test)
        rf_predictions = rf_model.predict(X_test)

        accuracy = rf_model.score(X_test, y_test)
        print(f"Base model accuracy: {accuracy:.2f}")

        # Check the class distribution (should be ~45% home wins, ~25% draws, ~30% away wins)
        print(y_test.value_counts(normalize=True))

        # ========== OPTIMIZING HYPERPARAMETERS ==========
        print("\n========== TUNED MODEL ==========")
        param_grid = {
            "n_estimators": [50, 70, 90, 110, 130, 150],
            "min_samples_split": [2, 3, 4, 5, 7],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", None],
        }

        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_
        tuned_accuracy = best_model.score(X_test, y_test)
        print(f"Tuned model accuracy: {tuned_accuracy:.2f}")

        # ========== EVALUATE ==========
        tuned_predictions = best_model.predict(X_test)
        cm = confusion_matrix(y_test, tuned_predictions, labels=["H", "D", "A"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Home Win", "Draw", "Away Win"]
        )
        disp.plot()
        plt.title("Random Forest - Confusion Matrix")
        # plt.savefig("/evaluation/confusion_matrix.png")
        plt.show()

        print(
            classification_report(
                y_test, tuned_predictions, target_names=["Home Win", "Draw", "Away Win"]
            )
        )

        # ========== HYBRID PREDICTOR ==========
        hybrid = HybridPredictor(best_model, league_avg, rf_weight=0.8)
        hybrid_preds = hybrid.predict(X_test)

        hybrid_accuracy = (hybrid_preds == y_test).mean()
        print(f"Hybrid accuracy: {hybrid_accuracy:.2f}")
