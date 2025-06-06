import os

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from openml import OpenMLDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    """
    Class for Exploratory Data Analysis (EDA) on a dataset.
    """

    def __init__(self):
        """
        Initialize the EDA class.

        On initialization, assets folders are created if they don't exist, following Config class.
        """
        self.dataset: OpenMLDataset | None = None
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
        self.X, self.y, _, _ = self.dataset.get_data(target=self.dataset.default_target_attribute)

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

    def plot_visualization(self, file_name_suffix="") -> None:
        """
        Plot visualizations of the dataset.
        1. Histograms / KDE plots
        2. Correlation heatmap
        3. Class distribution (sns.countplot)
        """

        # Histograms
        self.X[sorted(self.X.columns)].hist(figsize=(20, 15), bins=100)
        plt.suptitle("Histograms of the dataset", fontsize=20)
        plt.tight_layout(pad=1)
        plt.savefig(Config.DIR_EDA + f"/histograms{file_name_suffix}.png")

        # Correlation heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(self.X[sorted(self.X.columns)].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation heatmap of the dataset", fontsize=20)
        plt.savefig(Config.DIR_EDA + f"/correlation_heatmap{file_name_suffix}.png")

        # Class distribution
        plt.figure(figsize=(20, 15))
        sns.countplot(x=self.y, data=self.X)
        plt.title("Class distribution of the dataset", fontsize=20)
        plt.xticks(rotation=90)
        plt.savefig(Config.DIR_EDA + f"/class_distribution{file_name_suffix}.png")

    def data_checks(self, file_name_suffix="") -> None:
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

        # Check for outliers using boxplots
        n_cols = 7
        n_rows = (len(self.X.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(self.X.columns):
            sns.boxplot(y=self.X[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(Config.DIR_EDA + f"/boxplots{file_name_suffix}.png")

        # Check for outliers using IQR

        lower_bound, upper_bound = self.outliers_iqr()
        outliers = ((self.X < lower_bound) | (self.X > upper_bound)).sum()
        print("----------------------------------")
        print("Number of outliers in the dataset:")
        print(outliers)
        print("----------------------------------")

    def outliers_iqr(self) -> (pd.Series, pd.Series):
        """
        Calculate the IQR and return the lower and upper bounds for outliers.
        """
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)

        IQR: pd.Series = Q3 - Q1

        lower_bound: pd.Series = Q1 - 1.5 * IQR
        upper_bound: pd.Series = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    def normalize_dataset(self) -> None:
        """
        Normalize the dataset using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        print("Dataset normalized using MinMaxScaler.")

    def standardize_dataset(self) -> None:
        """
        Standardize the dataset using StandardScaler.
        """
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        print("Dataset standardized using StandardScaler.")

    def remove_duplicates(self) -> None:
        """
        Remove duplicates lines from the dataset.
        """
        print("--------------------------------")
        print("Number of duplicate lines in the dataset:")
        print(self.X.duplicated().sum())
        print("--------------------------------")

        mask = self.X.duplicated(keep="first")
        self.X = self.X[~mask]
        self.y = self.y[~mask]
        print("Duplicates removed from the dataset.")

    def remove_outliers(self) -> None:
        """
        Remove outliers from the dataset using IQR.
        """
        lower_bound, upper_bound = self.outliers_iqr()

        # Remove outliers
        mask = ~((self.X < lower_bound) | (self.X > upper_bound)).any(axis=1)
        self.X = self.X.loc[mask]
        self.y = self.y.loc[mask]
        print("Outliers removed from the dataset.")

    def drop_highly_correlated_features(self) -> None:
        """
        Drop highly correlated features from the dataset.
        """
        to_drop = ['HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME']
        self.X = self.X.drop(columns=to_drop)
        print(f"Dropped highly correlated features: {to_drop}")

    def export_eda_members(self) -> (pd.DataFrame, pd.Series):
        """
        Export the dataset members (X and y).
        """
        return self.X, self.y
