import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train():
    # 1. Load Data
    # Ensure you have 'heart.csv' in the same folder
    if not os.path.exists('heart.csv'):
        print("Error: heart.csv not found. Please download from Kaggle.")
        return

    df = pd.read_csv('heart.csv')

    # 2. Preprocessing
    # Check for missing values (This dataset is usually clean, but good practice)
    df = df.dropna()
    
    # Split features and target
    # 'target' is the column name in this specific dataset (1 = disease, 0 = no disease)
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save Model
    joblib.dump(model, 'heart_model.pkl')
    print("\nModel saved as 'heart_model.pkl'")

if __name__ == "__main__":
    train()