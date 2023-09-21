import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def evaluate_model():
    # Load preprocessed data
    df = pd.read_csv("preprocessed_data.csv")
    
    # Define target and features
    X = df[['feature_1', 'feature_2', ...]]
    y = df['playoff']
    
    # Load the saved model
    clf = joblib.load("random_forest_model.pkl")
    
    # Make predictions
    y_pred = clf.predict(X)
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
