from numpy.f2py.symbolic import normalize

from classification_models import Classification_models
from eda import EDA
from train_test_model import Train_Test_Model

if __name__ == "__main__":
    ####################################################################################
    # Part 1 - EDA (Exploratory Data Analysis)
    ####################################################################################
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
    # Normalize and standardize numeric features

    # Exercise 7: Summarize the different insights of the EDA phase: Are there issues with some features?
    # High feature correlation ? Missing values ? Outliers ?

    # Exercise 8: Take the necessary action(s) and justify.

    # There is no missing values in the dataset.
    # Remove duplicates:
    eda.remove_duplicates()
    # Remove outlines:
    eda.remove_outliers()

    eda.plot_visualization(file_name_suffix="_post_duplicate_and_outliner_removal")
    eda.data_checks(file_name_suffix="_post_duplicate_and_outliner_removal")

    # Drop highly correlated features
    eda.drop_highly_correlated_features()

    eda.plot_visualization(file_name_suffix="_post_highly_correlated_features_removal")
    eda.data_checks(file_name_suffix="_post_highly_correlated_features_removal")

    # Save eda members
    X, y = eda.export_eda_members()

    ####################################################################################
    # Part 2: Train-Test Split and Baseline Model
    ####################################################################################
    train_test_model = Train_Test_Model(X, y)

    # Exercise 1: Split the dataset into train and test sets (80% train, 20% test)
    train_test_model.split_dataset()

    # Exercise 2: Train a baseline Decision Tree.
    # Exercise 3: Report accuracy on both train and test sets.
    train_test_model.train_baseline_decision_tree()

    # Exercise 4: Report precision, recall, F1 and the confusion matrix on the test set
    train_test_model.report_metrics()

    # Exercise 5: Any issue of overfitting ? Underfitting ? Other issues ?

    ####################################################################################
    # Part 3: Resampling with SMOTE
    ####################################################################################
    # Exercise 1: Based on the class distribution, do we need under-sampling or oversampling? Justify your choice.

    # Exercise 2: Apply SMOTE using the imblearn.over_sampling module.
    # Exercise 3: Show class distribution before and after.
    train_test_model.oversample_dataset()

    # Exercise 4: Is there any risk with SMOTE? Data leakage ?

    # Exercise 5: Are there other oversampling methods?

    X_train, y_train, X_test, y_test = train_test_model.export_members()
    ####################################################################################
    # Part 4: Classificaiton Models (7 exercices)
    ####################################################################################

    classification_model = Classification_models(X_train, y_train, X_test, y_test)
    # Exercise 1: Decision Tree (baseline)
    classification_model.decision_tree()

    # Exercise 2: Decision Tree with Pre-Pruning
    classification_model.decision_tree_with_pre_pruning()

    # Exercise 3: Post-Pruning
    classification_model.post_pruning()

    # Exercise 4: Naive Bayes
    classification_model.naive_bayes()

    # Exercise 5: Random Forest
    classification_model.random_forest()

    # Exercise 6: Boosting (AdaBoost, GradientBoosting and XGBoost)
    classification_model.boosting()

    # Exercise 7: Stacking