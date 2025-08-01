# src/train.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import load_data, evaluate_model, save_params
import os

def main():
    # Load dataset
    X, y = load_data()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    save_params(model, "models/model.joblib")

    # Evaluate on test set
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
