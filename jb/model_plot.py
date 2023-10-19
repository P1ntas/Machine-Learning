import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmaps(predictions_df):
    classifiers = predictions_df['Classifier'].unique()
    teams = predictions_df['tmID'].unique()
    years = predictions_df['Year'].unique()

    print("Heatmap data:")
    print(classifiers)
    print(teams)
    print(years)


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

#plot bar chart with one bar being the result another being the prediction

def plot_bar_chart(predictions_df):
    predictions_df['Year'] = predictions_df['Year'].astype(int)

    classifiers = predictions_df['Classifier'].unique()
    teams = predictions_df['tmID'].unique()
    years = predictions_df['Year'].unique()

    # Width of a bar 
    width = 0.35 

    # Create a single figure to plot all bar charts
    fig, axes = plt.subplots(nrows=len(classifiers), figsize=(15, 5*len(classifiers)))

    for idx, classifier in enumerate(classifiers):
        ax = axes[idx] if len(classifiers) > 1 else axes

        for team in teams:
            subset = predictions_df[(predictions_df['Classifier'] == classifier) & (predictions_df['tmID'] == team)]
            # Check if there's any data for the team for this classifier
            if not subset.empty:
                year_positions = range(len(subset['Year']))
                ax.bar([pos - width/2 for pos in year_positions], subset['Actual'], width, label=f'{team} Actual', alpha=0.5)
                ax.bar([pos + width/2 for pos in year_positions], subset['Predicted'], width, label=f'{team} Predicted', alpha=0.5)

        ax.set_title(classifier)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.set_xticks(year_positions)
        ax.set_xticklabels(subset['Year'])
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_line_chart(predictions_df):
    predictions_df['Year'] = predictions_df['Year'].astype(int)

    classifiers = predictions_df['Classifier'].unique()
    years = predictions_df['Year'].unique()
    
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
                accuracy_dict[classifier].append(None)  # Using None for years with no data

    # Create a single figure for plotting
    plt.figure(figsize=(15, 6))

    # Plot the accuracy for each classifier
    for classifier, accuracies in accuracy_dict.items():
        plt.plot(years, accuracies, label=classifier, marker='o')

    plt.title("Prediction Accuracy Over Years")
    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

