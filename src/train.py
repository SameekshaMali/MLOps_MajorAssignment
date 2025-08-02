# src/train.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import load_data, evaluate_model, save_params
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Load dataset
    X, y = load_data()
    print("Feature sample stats:", X.mean(), X.std())


    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("Trained Weights sample:", model.coef_[:3])


    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    save_params(model, "models/model.joblib")
    save_params(scaler, "models/scaler.joblib")

    # Evaluate on test set
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
