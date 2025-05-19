import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from config import Config
from xgboost import XGBClassifier


def create_assets_folder() -> None:
    """
    Create the assets folder if it doesn't exist.
    """

    if not os.path.exists(Config.DIR_ASSETS):
        os.makedirs(Config.DIR_ASSETS)

    if not os.path.exists(Config.DIR_MODELS):
        os.makedirs(Config.DIR_MODELS)


class Classification_models:
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model = None

        # Create the assets folder - if it doesn't exist
        create_assets_folder()

    def decision_tree(self) -> None:
        """
        Decision Tree Classifier
        """
        model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
        model = model.fit(self.X_train, self.y_train)

        self.model = model

        self.performance_evaluation()

    def performance_evaluation(self) -> None:
        """
        Report metrics (confusion matrix, precision, recall, F1)
        """
        y_pred = self.model.predict(self.X_test)

        print("----------------------------------")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, self.model.predict(self.X_test)))
        print("----------------------------------")

    def decision_tree_with_pre_pruning(self) -> None:
        """
        Decision Tree Classifier with Pre-Pruning
        Use GridSearchCV (with 5 folds) to tune max_depth, min_samples_split. Try with max_depth
        (3, 5, 10, 20, 30, None) and min_smaples_split (5, 10, 20, 30).
        Use “f1-macro” as the scoring metric for the grid search.
        """
        param_grid = {'max_depth': [3, 5, 10, 20, 30, None], 'min_samples_split': [5, 10, 20, 30]}

        model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)

        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_

        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)

        self.performance_evaluation()

    def post_pruning(self) -> None:
        """
        Use ccp_alpha pruning path and cost_complexity_pruning_path
        Compare results with baseline and with the pre-pruned tree
        """
        model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE)
        model = model.fit(self.X_train, self.y_train)

        path = model.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas = path.ccp_alphas[:-1]

        models = []
        for ccp_alpha in ccp_alphas:
            model = DecisionTreeClassifier(random_state=Config.RANDOM_STATE, ccp_alpha=ccp_alpha)
            model.fit(self.X_train, self.y_train)
            models.append(model)

        f1_scores = [
            f1_score(self.y_test, model.predict(self.X_test), average='macro') for model in models
        ]

        # Plot alpha vs F1-macro
        plt.figure(figsize=(10, 6))
        plt.plot(ccp_alphas, f1_scores, marker='o')
        plt.xlabel("ccp_alpha")
        plt.ylabel("F1 Macro Score on Test Set")
        plt.title("Post-Pruning: Alpha vs F1 Score")
        plt.grid(True)
        plt.savefig(f"{Config.DIR_MODELS}/post_pruning_alpha_vs_f1.png")

        # Choose best-performing model
        best_index = f1_scores.index(max(f1_scores))
        best_model = models[best_index]
        best_alpha = ccp_alphas[best_index]

        print(f"Best ccp_alpha: {best_alpha}")

        self.model = best_model

        self.performance_evaluation()

    def naive_bayes(self) -> None:
        """
        Use GaussianNB and/or MultinomialNB or (depending on feature preprocessing)
        Compare results with previous models
        """
        model = GaussianNB()
        model = model.fit(self.X_train, self.y_train)

        self.model = model
        self.performance_evaluation()

    def random_forest(self) -> None:
        """
        Use RandomForestClassifier (with n_estimators=100)
        Plot the feature importance and discuss the results
        Compare results with previous models
        """
        model = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        model = model.fit(self.X_train, self.y_train)

        self.model = model

        self.performance_evaluation()

        # Plot feature importance
        importances = model.feature_importances_
        feature_names = self.X_train.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances (Random Forest)")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{Config.DIR_MODELS}/random_forest_feature_importance.png")
        plt.close()

    def boosting(self) -> None:
        """
        Try AdaBoostClassifier (with n_estimators=100)
        GradientBoostingClassifier (with n_estimators=100)
        Compare training time
        Compare results with previous models
        """
        ada = AdaBoostClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        gboost = GradientBoostingClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        xgboost = XGBClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)

        # Train and evaluate each model
        ada_time = self.train_boost(ada)
        gboost_time = self.train_boost(gboost)
        xgboost_time = self.train_boost(xgboost)
        print(f"Training time for AdaBoost: {ada_time:.2f} seconds")
        print(f"Training time for GradientBoosting: {gboost_time:.2f} seconds")
        print(f"Training time for XGBoost: {xgboost_time:.2f} seconds")


    def train_boost(self, model) -> float:
        """
        Train the model and measure the time taken.
        """
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        self.model = model

        self.performance_evaluation()

        return training_time