import os

import matplotlib.pyplot as plt
import openml
import pandas as pd
import seaborn as sns

from config import Config


def create_assets_folder() -> None:
    """
    Create the assets folder if it doesn't exist.
    """

    if not os.path.exists(Config.DIR_ASSETS):
        os.makedirs(Config.DIR_ASSETS)

    if not os.path.exists(Config.DIR_EDA):
        os.makedirs(Config.DIR_EDA)


class EDA:
    def __init__(self):
        self.dataset = None
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series()

        # Create the assets folder - if it doesn't exist
        create_assets_folder()

    def load_dataset(self) -> None:
        """
        Load a dataset from OpenML.
        Source of dataset: OpenML - https://www.openml.org/search?type=data&status=active&id=1049&sort=runs
        """
        # Load the dataset from local repository
        self.dataset = openml.datasets.get_dataset(Config.DATASET_ID)
        self.X, self.y, _, _ = self.dataset.get_data()

    def summary_table(self) -> pd.DataFrame:
        """
        Return a summary table of the dataset.
        We show the different features by - Feature, Description
        """
        # Get the feature names and types
        feature_names = self.X.columns
        feature_types = self.X.dtypes

        df = pd.DataFrame(
            {
                "Feature Name": feature_names,
                "Type": feature_types,
                "Distinct Values": [self.X[col].nunique() for col in feature_names],
                "Missing Values": [self.X[col].isnull().sum() for col in feature_names],
            }
        ).sort_values(by="Feature Name")

        return df

    def summary_statictics(self) -> None:
        """
        Print summary statistics of the dataset.
        """
        print("----------------------------------")
        print("Summary statistics of the dataset:")
        print(self.X.describe().to_markdown())
        print("----------------------------------")

        print("----------------------------------")
        print("\nSummary statistics of the target variable:")
        print(self.X.info())
        print("----------------------------------")

        print("----------------------------------")
        print("\nFirst 5 rows of the dataset:")
        print(self.X.head())
        print("----------------------------------")


    def plot_visualization(self) -> None:
        """
        Plot visualizations of the dataset.
        1. Histograms / KDE plots
        2. Correlation heatmap
        3. Class distribution (sns.countplot)
        """
        # Histograms / KDE plots
        self.X.hist(figsize=(20, 15), bins=100)
        plt.suptitle("Histograms of the dataset", fontsize=20)
        plt.savefig(Config.DIR_EDA + "/histograms.png")
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(self.X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation heatmap of the dataset", fontsize=20)
        plt.savefig(Config.DIR_EDA + "/correlation_heatmap.png")
        plt.show()

        # Class distribution
        plt.figure(figsize=(20, 15))
        sns.countplot(x=self.y, data=self.X)
        plt.title("Class distribution of the dataset", fontsize=20)
        plt.xticks(rotation=90)
        plt.savefig(Config.DIR_EDA + "/class_distribution.png")
        plt.show()

    def data_checks(self) -> None:
        """
        Checks:
        1. Missing values (df.isnull().sum())
        2. Outliers (eg. boxplot, IQR)
        """
        # Check for missing values
        print("----------------------------------")
        print("Missing values in the dataset:")
        print(self.X.isnull().sum())
        print("----------------------------------")

        # Check for outliers using boxplot, display in separate subplots
        """self.X.boxplot(figsize=(20, 15))
        plt.suptitle("Boxplot of the dataset", fontsize=20)
        plt.xticks(rotation=90)
        plt.savefig(Config.DIR_EDA + "/boxplot.png")
        plt.show()"""

        n_cols = 7
        n_rows = (len(self.X.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(self.X.columns):
            sns.boxplot(y=self.X[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + "/boxplots.png")
        plt.show()