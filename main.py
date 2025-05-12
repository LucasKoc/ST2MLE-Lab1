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
