## MLOps Linear Regrssion Pipeline

## This Project implements an MLOps pipeline  with Linear Regression on the California Housing Dataset.

# Structure of the project folder
- src/ : training, quantization, prediction, utils
- tests/ : Unit Tests
- github/workflows : CI Pipeline

# Model Training 
The training script loads the California Housing Dataset, trains a `Linear Regression Model` using 
scikit-learn, evaluates it using R² and MSE, and saved the trained model to the '/models' directory.
Command used to run training:
python src/train.py
