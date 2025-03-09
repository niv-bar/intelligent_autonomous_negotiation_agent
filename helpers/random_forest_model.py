import pathlib
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Optional

from config.utils import load_config


class RandomForestRegressorModel:
    """
    Training pipeline for the Group14 Negotiation Agent using a Random Forest Regressor.

    Handles data loading, preprocessing, model training, and saving of trained components.
    """

    # File paths for saving the model and scaler
    MODEL_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_rf_model.pkl"
    SCALER_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "group14_rf_scaler.pkl"
    TRAINING_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "training_data"

    # Load configuration for features and target variable
    CONFIG = load_config(file_name="group14")

    def __init__(self, max_depth: int = 10, n_estimators: int = 100, min_samples_split: int = 5):
        """
        Initializes the training pipeline.

        Args:
            max_depth (int, optional): Maximum depth of the decision trees. Default is 10.
            n_estimators (int, optional): Number of trees in the forest. Default is 100.
            min_samples_split (int, optional): Minimum samples required to split a node. Default is 5.
        """
        self.scaler = StandardScaler()
        self.model: Optional[RandomForestRegressor] = None
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split

    def build_model(self) -> None:
        """
        Builds and initializes the Random Forest model.
        """
        self.model = RandomForestRegressor(
            random_state=42,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            n_jobs=-1  # Use all available CPU cores
        )

    def train_model(self, test_size: float = 0.2) -> None:
        """
        Trains the Random Forest regression model using the provided dataset.

        Args:
            test_size (float, optional): Proportion of data to be used for testing. Default is 0.2 (20%).
        """
        # Load and preprocess dataset
        all_scores_combined = pd.concat([
            pd.read_csv(self.TRAINING_PATH / f"all_scores_v{i}.csv")
            for i in range(1, 6)
        ], ignore_index=True)

        df = all_scores_combined[all_scores_combined['strategy'] != 'group14'].copy()

        # Drop rows with missing values
        df = df.dropna(subset=[self.CONFIG["target_feature"]] + self.CONFIG["ml_features"])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.CONFIG["ml_features"]], df[self.CONFIG["target_feature"]],
            test_size=test_size,
            random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train the model
        self.build_model()
        self.model.fit(X_train_scaled, y_train)

        # Save the trained model and scaler
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)

        print("Training completed. Model and scaler saved successfully.")


if __name__ == "__main__":
    RandomForestRegressorModel().train_model()
