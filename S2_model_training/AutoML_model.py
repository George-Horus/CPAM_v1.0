"""
===============================================================
Script Name: Automl_model.py

Script Overview
---------------------------------------------------------------
This script is an automated machine learning (AutoML) comparison
and modeling module designed for applications such as remote
sensing inversion, environmental monitoring, and agricultural
forecasting. It is primarily used for regression modeling tasks.

Key Features
---------------------------------------------------------------
✅ Supports multiple AutoML frameworks:
    - **AutoGluon**
        - Automated model search
        - Ensemble learning (bagging, stacking)
        - Automatic hyperparameter tuning
    - **TPOT**
        - Genetic algorithm-based automated pipeline search
        - Automatic feature selection and construction
        - Provides scikit-learn pipelines

✅ Automated training and evaluation:
    - Automatically splits training and test sets
    - Trains multiple AutoML models
    - Compares model performance automatically
    - Saves the best model automatically

✅ Multiple regression evaluation metrics:
    - R²
    - RMSE
    - MAE
    - MSE
    - Pearson correlation coefficient

✅ Visualization:
    - Generates scatter plots comparing true vs. predicted values
    - Outputs PNG images suitable for papers or reports

✅ Automatic saving:
    - Automatically saves the best model to the specified directory
    - AutoGluon automatically saves the full training process
    - TPOT pipeline is saved as a `.pkl` file

Input Data
---------------------------------------------------------------
- Excel or DataFrame format:
    - Column 1: target variable (e.g. nitrogen content, chlorophyll content, yield)
    - From column 2 onward: input features (band values, vegetation indices, etc.)

Typical Workflow
---------------------------------------------------------------
from automl_training_pipeline import AutoML_training_main

# Load data
data = pd.read_excel("best_indices.xlsx")

# Define model save path
save_dir = r"D:\\YourPath\\ModelOutput"

# Automatically train and compare models
fig, best_model_name, metrics, model_path = AutoML_training_main(data, save_dir)

print(f"Best model: {best_model_name}")
print(metrics)

# Show comparison figure
fig.show()

Output Results
---------------------------------------------------------------
- Name of the best model (AutoGluon or TPOT)
- Dictionary of model performance metrics
- File path of the best model
- PNG figure comparing AutoML models

Use Cases
---------------------------------------------------------------
- Remote sensing inversion
- Crop yield prediction
- Water quality monitoring
- Environmental variable prediction
- Any regression task suitable for AutoML modeling

===============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from tpot import TPOTRegressor
import warnings
warnings.filterwarnings("ignore")
import os
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AutoMLComparators:
    def __init__(self, data, X_train, X_test, y_train, y_test, label_name):
        self.data = data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_name = label_name
        self.models = {}
        self.results = {}
        self.predictions = {}  # Store predictions from each model

    def train_autogluon(self, save_dir):
        """
        Train regression model using AutoGluon
        """
        train_df = pd.concat([self.y_train, self.X_train], axis=1)

        predictor = TabularPredictor(
            label=self.label_name,
            problem_type="regression",
            eval_metric="r2",
            verbosity=2,
            path=os.path.join(save_dir, "AutoGluon")
        )

        predictor.fit(
            train_df,
            time_limit=600,
            presets="best_quality",
            excluded_model_types=['NN_TORCH'],
            hyperparameters={
                "GBM": {
                    "extra_trees": True,
                    "ag_args": {"name_suffix": "XGB"},
                    "n_estimators": 500,
                    "learning_rate": 0.05
                },
                "CAT": {"iterations": 500},
                "RF": {"n_estimators": 200},
            },
            num_bag_folds=5,
            auto_stack=True,
        )

        y_pred = predictor.predict(self.X_test)
        self.models["AutoGluon"] = predictor
        self.predictions["AutoGluon"] = y_pred
        self.results["AutoGluon"] = self._evaluate(self.y_test, y_pred)

    def train_tpot(self):
        """
        Train TPOT model
        """
        tpot = TPOTRegressor(
            generations=3,
            population_size=20,
            max_time_mins=0.5,
            max_eval_time_mins=0.5,
            cv=3,
            n_jobs=1,
            random_state=42
        )
        tpot.fit(self.X_train, self.y_train)
        y_pred = tpot.predict(self.X_test)
        self.models["TPOT"] = tpot
        self.predictions["TPOT"] = y_pred
        self.results["TPOT"] = self._evaluate(self.y_test, y_pred)

    def _evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        return {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "Pearson": pearson
        }

    def plot_comparison(self, title="Comparison of AutoML Model Predictions"):
        """
        Plot comparison of prediction results from different models
        """
        import seaborn as sns
        sns.set_theme(style="whitegrid")

        num_models = len(self.predictions)
        cols = 2
        rows = 1

        plt.figure(figsize=(12, 5))

        for i, (name, y_pred) in enumerate(self.predictions.items(), 1):
            r2 = r2_score(self.y_test, y_pred)

            plt.subplot(rows, cols, i)
            plt.scatter(
                self.y_test,
                y_pred,
                alpha=0.7,
                edgecolors='k',
                s=40,
                label="Predictions"
            )
            plt.plot(
                [min(self.y_test), max(self.y_test)],
                [min(self.y_test), max(self.y_test)],
                'r--',
                linewidth=1.5
            )
            plt.title(f"{name} (R²: {r2:.2f})", fontsize=14, fontweight='bold')
            plt.xlabel("True Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.legend(fontsize=10)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        return plt.gcf()

def AutoML_training_main(data: pd.DataFrame, save_dir: str):
    """
    Main function for automated machine learning training
    """

    # === Ensure working directory and save_dir are on the same drive ===
    drive_letter = os.path.splitdrive(save_dir)[0]
    os.chdir(drive_letter + "\\")

    os.makedirs(save_dir, exist_ok=True)

    label_name = data.columns[0]
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    comparator = AutoMLComparators(data, X_train, X_test, y_train, y_test, label_name)
    comparator.train_autogluon(save_dir)
    comparator.train_tpot()

    best_model_name, best_metrics = max(
        comparator.results.items(), key=lambda x: x[1]['R2']
    )
    best_model = comparator.models[best_model_name]

    if best_model_name == "AutoGluon":
        model_path = os.path.join(save_dir, "AutoGluon")
        # No need to save separately; AutoGluon saves during fit
    elif best_model_name == "TPOT":
        model_path = os.path.join(save_dir, "TPOT.pkl")
        joblib.dump(best_model.fitted_pipeline_, model_path)

    print(f"Best AutoML model: {best_model_name}, R²: {best_metrics['R2']:.4f}")
    print(f"Best AutoML model saved at: {model_path}")

    metrics = comparator.results

    # Plot
    AutoML_comparison_scatter_fig = comparator.plot_comparison()
    fig_path = os.path.join(save_dir, "automl_model_comparison.png")
    if AutoML_comparison_scatter_fig:
        AutoML_comparison_scatter_fig.savefig(fig_path, dpi=300)
        print(f"AutoML model comparison figure saved at: {fig_path}")

    return AutoML_comparison_scatter_fig, best_model_name, metrics, model_path

if __name__ == "__main__":
    file_path = r'best_indices1.xlsx'
    res_df = pd.read_excel(file_path, sheet_name='Sheet1')
    save_dir = r'D:\Code_Store\Water_content_inversion\model_train\Best_Model'
    AutoML_comparison_scatter_fig, best_model_name, metrics, model_path = AutoML_training_main(
        res_df, save_dir
    )
    print(metrics)
    AutoML_comparison_scatter_fig.show()
