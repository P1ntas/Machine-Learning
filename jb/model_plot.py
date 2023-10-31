import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    try:
        predictions_df = pd.read_csv(file_path)
        predictions_df['Year'] = predictions_df['Year'].astype(int)  # Ensure Year is int type
        return predictions_df
    except Exception as e:
        logging.error("Error loading the CSV file: ", e)
        return None

def plot_heatmaps(predictions_df):
    classifiers = predictions_df['Classifier'].unique()
    teams = predictions_df['tmID'].unique()
    years = sorted(predictions_df['Year'].unique())

    print("Heatmap data:")
    print("Classifiers:", classifiers)
    print("Teams:", teams)
    print("Years:", years)

    # Create a single figure to plot all heatmaps
    fig, axes = plt.subplots(nrows=len(classifiers), figsize=(15, 5*len(classifiers)))

    for idx, classifier in enumerate(classifiers):
        # Create an empty dataframe to store probabilities
        heatmap_data = pd.DataFrame(index=teams, columns=years)
        heatmap_data.fillna(0, inplace=True)

        # Fill in the dataframe with probabilities
        subset = predictions_df[predictions_df['Classifier'] == classifier]
        for _, row in subset.iterrows():
            heatmap_data.at[row['tmID'], row['Year']] = float(row['Probability'])

        # Plot the heatmap
        ax = axes[idx] if len(classifiers) > 1 else axes
        sns.heatmap(heatmap_data, cmap="PuBu", ax=ax, annot=True)
        ax.set_title(classifier)
        ax.set_xlabel("Year")
        ax.set_ylabel("Team ID")

    plt.tight_layout()
    plt.show()

def plot_bar_chart(predictions_df):
    classifiers = predictions_df['Classifier'].unique()
    years = sorted(predictions_df['Year'].unique())

    # Width of a bar 
    width = 0.35 

    # Create a single figure to plot all bar charts
    fig, axes = plt.subplots(nrows=len(classifiers), figsize=(15, 5*len(classifiers)))
    
    if len(classifiers) == 1:
        axes = [axes]

    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]
        
        correct = []
        incorrect = []

        for year in years:
            subset = predictions_df[(predictions_df['Classifier'] == classifier) & (predictions_df['Year'] == year)]
            if not subset.empty:
                correct_count = sum(subset['Predicted'] == subset['Actual'])
                incorrect_count = len(subset) - correct_count
                correct.append(correct_count)
                incorrect.append(incorrect_count)
            else:
                correct.append(0)
                incorrect.append(0)

        # Ensure we have data to plot
        if not correct and not incorrect:
            logging.warning(f"No data available for classifier {classifier}.")
            continue

        # Positions of the bar on X-axis
        ind = np.arange(len(years))

        ax.bar(ind, correct, width, label='Correct', alpha=0.7)
        ax.bar(ind, incorrect, width, bottom=correct, label='Incorrect', alpha=0.7)

        ax.set_title(classifier)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Predictions")
        ax.set_xticks(ind)
        ax.set_xticklabels(years)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_line_chart(predictions_df):
    predictions_df['Year'] = predictions_df['Year'].astype(int)

    classifiers = predictions_df['Classifier'].unique()
    years = sorted(predictions_df['Year'].unique())
    
    # Create a dictionary to store accuracy per classifier per year
    accuracy_dict = {classifier: [] for classifier in classifiers}

    for year in years:
        for classifier in classifiers:
            subset = predictions_df[(predictions_df['Classifier'] == classifier) & (predictions_df['Year'] == year)]
            if not subset.empty:
                correct_predictions = sum(subset['Predicted'] == subset['Actual'])
                total_predictions = len(subset)
                accuracy = correct_predictions / total_predictions
                accuracy_dict[classifier].append(accuracy)
            else:
                accuracy_dict[classifier].append(np.nan)  # Using np.nan for years with no data

    # Determine the number of subplots needed
    num_subplots = len(classifiers)
    
    # Create subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(15, 6 * num_subplots))

    # If there's only one classifier, axs is not a list, so we wrap it in a list
    if num_subplots == 1:
        axs = [axs]

    # Plot the accuracy for each classifier
    for idx, (classifier, accuracies) in enumerate(accuracy_dict.items()):
        if any(~np.isnan(accuracies)):  # Check if there is any data available for this classifier
            axs[idx].plot(years, accuracies, marker='o')
            axs[idx].set_title(f"{classifier} Prediction Accuracy Over Years")
            axs[idx].set_xlabel("Year")
            axs[idx].set_ylabel("Accuracy")
            axs[idx].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "jb/predictions_results-playerTeams.csv"
    predictions_df = load_data(file_path)
    
    if predictions_df is not None:
        plot_heatmaps(predictions_df)
        plot_bar_chart(predictions_df)
        plot_line_chart(predictions_df)
