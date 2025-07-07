"""
===============================================================
Script Name: ML_model.py

Script Overview
---------------------------------------------------------------
This script is a traditional machine learning (ML) pipeline
for multi-model comparison and regression modeling. It is
designed for high-dimensional data scenarios such as remote
sensing inversion, agricultural prediction, and environmental
monitoring.

Key Features
---------------------------------------------------------------
✅ Integrates multiple regression models:
    - SVR (Support Vector Regression)
    - MLR (Multiple Linear Regression)
    - RandomForest
    - KNN (K-Nearest Neighbors Regression)
    - GBDT (Gradient Boosted Decision Trees)
    - PLSR (Partial Least Squares Regression)
    - XGBoost
    - ELM (Extreme Learning Machine, custom implementation)
    - BPNN (Multi-Layer Perceptron Regression)

✅ Automated training and comparison:
    - Automatically splits training and test sets
    - Trains multiple regression models
    - Evaluates model performance
    - Automatically selects the best model based on R²
    - Automatically saves the best model

✅ Rich regression metrics:
    - R²
    - RMSE
    - MAE
    - MSE
    - Pearson correlation coefficient

✅ Visualization:
    - Generates scatter plots of true vs. predicted values
    - Compares multiple models side by side
    - Suitable for reports and publications

✅ Automatic saving:
    - Models saved as .joblib files
    - Comparison plots output as PNG

Input Data Format
---------------------------------------------------------------
- Excel file or DataFrame:
    - Column 1: target variable (e.g. nitrogen content, yield, water content, etc.)
    - From column 2 onward: input features (band values, vegetation indices, environmental factors, etc.)

Output Results
---------------------------------------------------------------
- Name of the best model
- Dictionary of performance metrics for all models
- Path to the saved best model file
- Scatter plot comparing model predictions (PNG)

Notes
---------------------------------------------------------------
- This script does not perform normalization on the input data.
- ELM is a custom implementation of the Extreme Learning Machine model.
- For small datasets, some models may overfit or perform unstably.
- This pipeline is designed for regression tasks and does not support classification.

===============================================================
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from numpy.linalg import pinv
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

class ELMRegressor:
    def __init__(self, n_hidden=20, random_state=None):
        self.n_hidden = n_hidden
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)
        self.input_weights = np.random.normal(size=[X.shape[1], self.n_hidden])
        self.biases = np.random.normal(size=[self.n_hidden])
        H = np.tanh(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(pinv(H), y)

    def predict(self, X):
        H = np.tanh(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)

class MLModels:
    def __init__(self):
        self.models = {
            'SVR': SVR(),
            'MLR': LinearRegression(),
            'RandomForest': RandomForestRegressor(),
            'KNN': KNeighborsRegressor(),
            'GBDT': GradientBoostingRegressor(),
            'PLSR': PLSRegression(n_components=2),
            'XGBoost': xgb.XGBRegressor(),
            'ELM': ELMRegressor(n_hidden=50),
            'BPNN': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
        }

    def training(self, X_train, X_test, Y_train, Y_test):
        predictions = {}
        r2_scores = {}
        for name, model in self.models.items():
            model.fit(X_train, Y_train)
            predictions[name] = model.predict(X_test)
            r2_scores[name] = r2_score(Y_test, predictions[name])
        return predictions, r2_scores

    def evaluate_models(self, X_train, X_test, Y_train, Y_test):
        metrics = {}
        for name, model in self.models.items():
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(Y_test, predictions)
            rmse = np.sqrt(mean_squared_error(Y_test, predictions))
            mse = mean_squared_error(Y_test, predictions)
            mae = mean_absolute_error(Y_test, predictions)
            pearson_corr, _ = pearsonr(Y_test, predictions)
            metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'Pearson': pearson_corr
            }
        return metrics

    def select_best_model(self, metrics):
        return max(metrics.items(), key=lambda item: item[1]['R2'])

    def ML_model_comparison_scatter(self, X_train, X_test, Y_train, Y_test, title='Comparison of ML Model Predictions'):
        predictions, r2_scores = self.training(X_train, X_test, Y_train, Y_test)

        Y_test_original = Y_test

        plt.figure(figsize=(16, 12))
        sns.set_theme(style='whitegrid')

        for i, (name, prediction) in enumerate(predictions.items(), 1):
            plt.subplot(3, 3, i)
            plt.scatter(
                Y_test_original, prediction,
                alpha=0.7, edgecolors='k', s=40, label='Predictions'
            )
            plt.plot(
                [Y_test_original.min(), Y_test_original.max()],
                [Y_test_original.min(), Y_test_original.max()],
                'r--', linewidth=1.5
            )
            plt.title(
                f"{name} (R²: {r2_scores[name]:.2f})",
                fontsize=14, fontweight='bold'
            )
            plt.xlabel("True Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.legend(fontsize=10)

        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return plt.gcf()

def ML_training_main(data, save_dir):
    """
    Trains multiple models directly without performing normalization
    and outputs information about the best model.
    """
    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    model_instance = MLModels()
    metrics = model_instance.evaluate_models(X_train, X_test, Y_train, Y_test)
    best_model_name, best_model_metrics = model_instance.select_best_model(metrics)

    os.makedirs(save_dir, exist_ok=True)
    model_file_path = os.path.join(
        save_dir, f"{best_model_name}_best_ML_model.joblib"
    )
    joblib.dump(model_instance.models[best_model_name], model_file_path)

    print(
        f"Best model: {best_model_name}, R2: {best_model_metrics['R2']:.4f}, "
        f"RMSE: {best_model_metrics['RMSE']:.4f}"
    )
    print(f"Model saved at: {model_file_path}")

    # Generate comparison plot
    fig = model_instance.ML_model_comparison_scatter(
        X_train, X_test, Y_train, Y_test
    )
    fig_path = os.path.join(save_dir, "ML_model_comparison_scatter.png")
    fig.savefig(fig_path, dpi=300)
    print(f"Model comparison plot saved at: {fig_path}")

    return fig, best_model_name, metrics, model_file_path
