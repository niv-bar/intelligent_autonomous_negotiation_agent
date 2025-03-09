import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
from typing import Optional

from config.utils import load_config


class NeuralNetwork(nn.Module):
    """
    Deep Learning model for regression-based negotiation strategy prediction.

    This neural network consists of three hidden layers with batch normalization,
    LeakyReLU activations, and dropout for regularization.
    """

    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        """
        Initializes the neural network architecture.

        Args:
            input_size (int): Number of input features.
            dropout_rate (float, optional): Dropout probability for regularization. Default is 0.2.
        """
        super(NeuralNetwork, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Predicted output value.
        """
        x = self.batch_norm1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)


class TrainingPipeline:
    """
    Training pipeline for the Group14 Negotiation Agent using a deep learning model.

    Handles data loading, preprocessing, model training, and saving of trained components.
    """

    # File paths for saving the model and scaler
    MODEL_PATH = pathlib.Path(__file__).parent.parent / "data" / "models" / "group14_nn_model.pth"
    SCALER_PATH = pathlib.Path(__file__).parent.parent / "data" / "models" / "group14_nn_scaler.pkl"
    TRAINING_PATH = pathlib.Path(__file__).parent.parent / "data" / "training_data"

    # Load configuration for features and target variable
    CONFIG = load_config(file_name="group14")

    def __init__(self, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Initializes the training pipeline.

        Args:
            batch_size (int, optional): Number of samples per batch during training. Default is 32.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model: Optional[NeuralNetwork] = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers

    def build_model(self, input_size: int) -> None:
        """
        Builds and initializes the deep learning model.

        Args:
            input_size (int): Number of input features.
        """
        self.model = NeuralNetwork(input_size)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    def train_model(self, epochs: int = 100, early_stopping_patience: int = 10) -> None:
        """
        Trains the model using the provided dataset.

        Implements early stopping and learning rate scheduling to optimize training.

        Args:
            epochs (int, optional): Number of training epochs. Default is 100.
            early_stopping_patience (int, optional): Number of epochs to wait before early stopping. Default is 10.
        """
        # Load and preprocess dataset
        all_scores_combined = pd.concat([
            pd.read_csv(f"{self.TRAINING_PATH}/all_scores_v{i}.csv")
            for i in range(1, 6)
        ], ignore_index=True)

        df = all_scores_combined[all_scores_combined['strategy'] != 'group14'].copy()

        # Drop rows with missing values
        df = df.dropna(subset=[self.CONFIG["target_feature"]] + self.CONFIG["ml_features"])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.CONFIG["ml_features"]], df[self.CONFIG["target_feature"]],
            test_size=0.2,
            random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.values).reshape(-1, 1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test.values).reshape(-1, 1)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Initialize model
        self.build_model(len(self.CONFIG["ml_features"]))

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            # Learning rate adjustment
            self.scheduler.step(avg_val_loss)

            # Early stopping mechanism
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.MODEL_PATH)  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save scaler for later inference
        joblib.dump(self.scaler, self.SCALER_PATH)
        print("Training completed. Model and scaler saved successfully.")


if __name__ == "__main__":
    TrainingPipeline().train_model()
