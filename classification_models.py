import pandas as pd


class Classification_models:
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model = None

    def decision_tree(self) -> None:
        """
        Decision Tree Classifier
        """
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        model = model.fit(self.X_train, self.y_train)

        # Performance Evaluation
        train_accuracy = accuracy_score(self.y_train, model.predict(self.X_train))
        test_accuracy = accuracy_score(self.y_test, model.predict(self.X_test))

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")

        self.model = model
