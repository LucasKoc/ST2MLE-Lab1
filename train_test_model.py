import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import Config


class Train_Test_Model:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None) -> None:
        self.X: pd.DataFrame = X if X is not None else pd.DataFrame()
        self.y: pd.Series = y if y is not None else pd.Series()
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series()
        self.y_test: pd.Series = pd.Series()

        self.model = None

    def split_dataset(
        self, test_size: float = Config.TEST_SIZE, random_state: int = Config.RANDOM_STATE
    ) -> None:
        """
        Split the dataset into train and test sets (80-20).
        Use stratified sampling to ensure that the class distribution is preserved in both sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print("Dataset split into train and test sets.")

    def train_baseline_decision_tree(self) -> None:
        """
        Train a baseline Decision Tree classifier on the training set and evaluate its performance on the test set.
        """
        model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
        model = model.fit(self.X_train, self.y_train)

        # Performance Evaluation
        train_accuracy = accuracy_score(self.y_train, model.predict(self.X_train))
        test_accuracy = accuracy_score(self.y_test, model.predict(self.X_test))

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")

        self.model = model

    def report_metrics(self) -> None:
        """
        Report precision, recall, F1 and the confusion matrix on the test set.
        """
        y_pred = self.model.predict(self.X_test)

        print("----------------------------------")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("----------------------------------")

        print("----------------------------------")
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("----------------------------------")

    def oversample_dataset(self) -> None:
        """
        Use SMOTE to oversample the minority class in the training set.
        """
        print("----------------------------------")
        print("New class distribution in the training set:")
        print("Before", self.y_train.value_counts())

        smote = SMOTE(random_state=Config.RANDOM_STATE)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        print("After", self.y_train.value_counts())
        print("----------------------------------")

    def export_members(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """
        Export the members of the class.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
