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
    df = df.select_dtypes(include=[float, int, bool])
    return df

def get_columns_to_remove(df):
    remove_columns = ['playoff', 'W', 'L', 'rank']
    remove_columns.extend([col for col in df.columns if 'Post' in col or 'post' in col])
    return remove_columns

def train_and_evaluate(df, remove_columns, years, i, classifier):
    train_years = years[:i]  # Use all years up to i
    test_year = years[i]
    
    logging.info(f"Training years: {train_years}")
    logging.info(f"Testing year: {test_year}")
    
    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    
    X_train = train.drop(remove_columns, axis=1)
    y_train = train['playoff']
    X_test = test.drop(remove_columns, axis=1)
    y_test = test['playoff']

    clf = classifier
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions)

def plot_results(years, results_dict):
    for classifier, results in results_dict.items():
        plt.plot(years, results, label=classifier)
    
    plt.xlabel('Year Predicted')
    plt.ylabel('Accuracy')
    plt.title('Rolling Window Results for Various Classifiers')
    plt.legend()
    plt.show()

def train_model():
    data_file_path = "preprocessed_data.csv"
    df = read_data(data_file_path)
    
    if df is not None:
        df = pre_process_data(df)
        remove_columns = get_columns_to_remove(df)
        
        years = df['year'].unique()
        years.sort()
        
        classifiers = {
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results_dict = {}

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(2, len(years)):
                accuracy = train_and_evaluate(df, remove_columns, years, i, classifier)
                logging.info(f"Classifier: {classifier_name}, Year: {years[i]}, Accuracy: {accuracy:.2f}")
                results.append(accuracy)
            results_dict[classifier_name] = results

        plot_results(years[2:], results_dict)

if __name__ == "__main__":
    train_model()
