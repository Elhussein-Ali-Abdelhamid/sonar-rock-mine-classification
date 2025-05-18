"""
Sonar Rock vs Mine Prediction
Author: Your Name
Date: YYYY-MM-DD

This script uses logistic regression to classify sonar signals as rocks or mines.
Dataset: sonar.all-data.csv (from UCI Machine Learning Repository)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # For model saving

def load_data(filepath):
    """Load and preprocess the dataset."""
    try:
        data = pd.read_csv("Copy of sonar data.csv", header=None)
        print("Dataset loaded successfully")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please check the path.")
        exit()

def train_model(X, y):
    """Train and evaluate the logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=1
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nModel Performance:")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.2%}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2%}")
    
    return model

def predict_sample(model, sample):
    """Make a prediction on a single sample."""
    sample_array = np.asarray(sample).reshape(1, -1)
    prediction = model.predict(sample_array)
    return prediction[0]

def main():
    # Load data
    data = load_data('sonar.all-data.csv')
    
    # Prepare features and target
    X = data.drop(60, axis=1)
    y = data[60]
    
    # Train model
    model = train_model(X, y)
    
    # Example prediction
    test_sample = (
        0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,
        0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,
        0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,
        0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,
        0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,
        0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055
    )
    
    prediction = predict_sample(model, test_sample)
    print(f"\nPrediction: {'Rock' if prediction == 'R' else 'Mine'}")

    # Save model for future use
    joblib.dump(model, 'sonar_model.pkl')
    print("\nModel saved as 'sonar_model.pkl'")

if __name__ == "__main__":
    main()