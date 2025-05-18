from eda import EDA

if __name__ == "__main__":
    # Part 1 - EDA (Exploratory Data Analysis)
    eda = EDA()

    # Exercise 1: Load the dataset
    eda.load_dataset()

    # Exercise 2: Create a summary table for the different features (Feature, Description)
    df_summary_table = eda.summary_table()
    print(df_summary_table.to_markdown(index=False))

    # Exercise 3: Summary statistics with df.describe().
    eda.summary_statictics()

    # Exercise 4: Visualize - Histograms / KDE plots + Correlation heatmap + Class distribution (sns.countplot)
    eda.plot_visualization()

    # Exercise 5: Check for - Missing values (df.isnull().sum()) + Outliers (eg. boxplot, IQR)
    eda.data_checks()

    # Exercise 6: Normalize and standardize numeric features (if you think it is needed)
    # It's needed here because the features are not on the same scale

    # Exercise 7: Summarize the different insights of the EDA phase: Are there issues with some features?
    # High feature correlation ? Missing values ? Outliers ?

    # Exercise 8: Take the necessary action(s) and justify.

    # There is no missing values in the dataset.
    # Remove duplicates:
    eda.remove_duplicates()

    # Remove outliers:

