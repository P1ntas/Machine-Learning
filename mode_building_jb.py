import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    return None

def pre_process_data(df):
    mapping = {'Y': 1, 'N': 0}
    df['playoff'] = df['playoff'].map(mapping)
    df.fillna(0, inplace=True)
    return df

def get_columns_to_remove():
    # Keep 'tmID' and do not remove it in this function.
    return ['playoff', 'rank', 'seeded', 'firstRound', 'semis', 'finals', 'lgID', 'franchID', 'confID', 'divID', 'name', 'arena']


def train_and_evaluate(df, years, i, classifier):
    train_years = years[:i]
    test_year = years[i]
    
    logging.info(f"Training years: {train_years}")
    logging.info(f"Testing year: {test_year}")
    
    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    
    remove_columns = get_columns_to_remove()
    
    X_train = train.drop(remove_columns + ['tmID'], axis=1)  # Removing 'tmID' for training.
    X_test = test.drop(remove_columns, axis=1)  # Keeping 'tmID' to merge later.
    y_train = train['playoff']
    y_test = test['playoff']

    clf = classifier
    clf.fit(X_train, y_train)
    
    # Temporarily removing 'tmID' for prediction.
    predictions = clf.predict(X_test.drop(['tmID'], axis=1))
    proba = clf.predict_proba(X_test.drop(['tmID'], axis=1))[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    
    # Returning 'tmID' to link predictions with teams.
    return predictions, proba, y_test.to_numpy(), X_test['tmID'].to_numpy(), accuracy


def plot_results(years, results_dict):
    for classifier, results in results_dict.items():
        plt.plot(years, results, label=classifier)
    
    plt.xlabel('Year Predicted')
    plt.ylabel('Accuracy')
    plt.title('Rolling Window Results for Various Classifiers')
    plt.legend()
    plt.show()

def train_model():
    data_file_path = "./basketballPlayoffs/teams.csv"
    df = read_data(data_file_path)
    
    if df is not None:
        df = pre_process_data(df)
        
        years = df['year'].unique()
        years.sort()
        
        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        
        results_dict = {}
        prediction_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(1, len(years)):
                predicted, predicted_proba, actual, tmIDs, accuracy = train_and_evaluate(df, years, i, classifier)

                # Add tmID to prediction_data
                for p, pp, a, tmID in zip(predicted, predicted_proba, actual, tmIDs):
                    prediction_data.append({
                        'tmID': tmID,
                        'Year': years[i],
                        'Classifier': classifier_name,
                        'Predicted': p,
                        'Probability': pp,
                        'Actual': a
                    })
                
                logging.info(f"Classifier: {classifier_name}, Year: {years[i]}, Accuracy: {accuracy:.2f}")
                results.append(accuracy)
            results_dict[classifier_name] = results
        
        plot_results(years[1:], results_dict)

        # Saving results to CSV
        predictions_df = pd.DataFrame(prediction_data)
        predictions_df.to_csv('predictions_results.csv', index=False)

if __name__ == "__main__":
    train_model()