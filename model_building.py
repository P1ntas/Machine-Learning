import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Load preprocessed data
    df = pd.read_csv("preprocessed_data.csv")
    
    # Define target and features
    X = df[['feature_1', 'feature_2', ...]]
    y = df['playoff']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and train the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(clf, "random_forest_model.pkl")

if __name__ == "__main__":
    train_model()
