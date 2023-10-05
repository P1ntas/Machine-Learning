import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train_model():
    df = pd.read_csv("preprocessed_data.csv")

  
    df['playoff'] = df['playoff'].apply(lambda x: 1 if x == 'Y' else 0)

    df.fillna(0, inplace=True)    
    
    df = df.select_dtypes(include=[float, int, bool])



    clf = DecisionTreeClassifier()
    results = []

   
    # Sliding Window
    years = df['year'].unique()
    years.sort()
    print(years)
    print(len(years))
    remove_columns = ['playoff', 'W', 'L', 'rank']
    for col in df.columns:
        if 'Post' in col or 'post' in col:
            remove_columns.append(col)
    print(remove_columns)
        
    for i in range(5, len(years)):
        print(years[:i])
        train_years = years[:i]
        test_year = years[i]
        
        train = df[df['year'].isin(train_years)]
        test = df[df['year'] == test_year]
        
        X_train = train.drop(remove_columns, axis=1)
        y_train = train['playoff']
        X_test = test.drop(remove_columns, axis=1)
        y_test = test['playoff']

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        print(predictions)
        accuracy = accuracy_score(y_test, predictions)
        results.append(accuracy)

    # Plot the results
    print(years[5:])
    plt.plot(years[5:], results)
    plt.xlabel('Year Predicted')
    plt.ylabel('Accuracy')
    plt.title('Sliding Window Results for Random Forest')
    plt.show()

if __name__ == "__main__":
    train_model()
