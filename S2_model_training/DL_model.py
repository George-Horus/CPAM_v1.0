"""
===============================================================
Script Name: DL_model.py

Script Overview
---------------------------------------------------------------
This script is an automated Deep Learning (DL) training and
comparison module designed for regression modeling scenarios,
particularly applicable in remote sensing inversion, agricultural
forecasting, water quality monitoring, and similar fields.

Key Features
---------------------------------------------------------------
✅ Supports multiple deep learning regression models:
    - **DNN (Deep Neural Network)**
    - **MLP (Multi-Layer Perceptron)**
    - **CNN (Convolutional Neural Network)**
    - **Autoencoder Regression**

✅ Automated workflow:
    - Automatically trains multiple DL models
    - Automatically compares model performance
    - Automatically selects the model with the best R²
    - Automatically saves the best model to the specified path
    - Automatically generates comparative visualization charts

✅ Rich performance metrics:
    - R²
    - RMSE
    - MAE
    - Pearson correlation coefficient

✅ High-quality visualization:
    - Generates scatter plots comparing true vs. predicted values
    - Supports comparison of multiple models
    - Automatically saves PNG images (suitable for publications or reports)

Input Data Format
---------------------------------------------------------------
- DataFrame or Excel file:
    - Column 1: target variable (e.g. nitrogen content, water content, yield, etc.)
    - Columns from 2 onward: feature data (e.g. band values, vegetation indices, etc.)

Typical Workflow
---------------------------------------------------------------
from dl_training_pipeline import DL_training_main

# Load data
data = pd.read_excel("best_features.xlsx")

# Specify model save path
save_dir = r"D:\\YourFolder\\Best_Model"

# Train and compare deep learning models
fig, best_model_name, metrics, model_file_path = DL_training_main(data, save_dir)

print(f"Best model: {best_model_name}")
print(metrics)

# Display comparison figure
fig.show()

Output Results
---------------------------------------------------------------
- Name of the best model
- Dictionary of performance metrics for all models
- File path of the best model (saved as .keras)
- Comparison scatter plot of DL model predictions (PNG)

Use Cases
---------------------------------------------------------------
- Remote sensing inversion (e.g. predicting nitrogen or water content)
- Crop yield prediction
- Environmental indicator prediction
- Exploring deep learning solutions for high-dimensional regression tasks

Notes
---------------------------------------------------------------
- The current version does not include data normalization or standardization.
- CNN input data automatically adds an extra dimension ([..., np.newaxis]).
- EarlyStopping is used to prevent overfitting.
- For small datasets, overfitting or instability may occur.

===============================================================
"""

import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

class DLModels:
    def __init__(self, X_train, Y_train):
        self.models = {
            'DNN': self.create_dnn(X_train, Y_train),
            'MLP': self.create_mlp(X_train, Y_train),
            'CNN': self.create_cnn(X_train, Y_train),
            'Autoencoder': self.create_autoencoder(X_train, Y_train)
        }

    def create_dnn(self, X_train, Y_train):
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, Y_train, epochs=100, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=10)])
        return model

    def create_mlp(self, X_train, Y_train):
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, Y_train, epochs=100, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=10)])
        return model

    def create_cnn(self, X_train, Y_train):
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            Conv1D(32, kernel_size=2, activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train[..., np.newaxis], Y_train, epochs=100, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=10)])
        return model

    def create_autoencoder(self, X_train, Y_train):
        input_layer = Input(shape=(X_train.shape[1],))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        output = Dense(1)(decoded)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, Y_train, epochs=100, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=10)])
        return model

    def evaluate_models(self, X_test, Y_test):
        metrics = {}
        for name, model in self.models.items():
            pred = model.predict(X_test[..., np.newaxis] if name == 'CNN' else X_test, verbose=0).flatten()
            r2 = r2_score(Y_test, pred)
            rmse = np.sqrt(mean_squared_error(Y_test, pred))
            mae = mean_squared_error(Y_test, pred)
            pearson_corr, _ = pearsonr(Y_test, pred)
            metrics[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Pearson': pearson_corr
            }
        return metrics

    def select_best_model(self, metrics):
        return max(metrics.items(), key=lambda item: item[1]['R2'])

    def DL_model_comparison_scatter(self, X_train, X_test, Y_train, Y_test):
        predictions = {}
        r2_scores = {}
        for name, model in self.models.items():
            pred = model.predict(X_test[..., np.newaxis] if name == 'CNN' else X_test, verbose=0).flatten()
            predictions[name] = pred
            r2_scores[name] = r2_score(Y_test, pred)

        plt.figure(figsize=(12, 10))
        sns.set_theme(style='whitegrid')

        for i, (name, prediction) in enumerate(predictions.items(), 1):
            plt.subplot(2, 2, i)
            plt.scatter(Y_test, prediction, alpha=0.7, edgecolors='k', s=40, label='Predictions')
            plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', linewidth=1.5)
            plt.title(f"{name} (R²: {r2_scores[name]:.2f})", fontsize=14, fontweight='bold')
            plt.xlabel("True Values", fontsize=12)
            plt.ylabel("Predicted Values", fontsize=12)
            plt.legend(fontsize=10)

        plt.suptitle("Comparison of DL Model Predictions", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return plt.gcf()

def DL_training_main(data, save_dir):
    X = data.iloc[:, 1:].values
    Y = data.iloc[:, 0].values

    # ❌ No normalization applied

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model_instance = DLModels(X_train, Y_train)
    metrics = model_instance.evaluate_models(X_test, Y_test)
    best_model_name, best_model_metrics = model_instance.select_best_model(metrics)

    os.makedirs(save_dir, exist_ok=True)
    model_file_path = os.path.join(save_dir, f"{best_model_name}_best_DL_model.keras")
    model_instance.models[best_model_name].save(model_file_path)

    print(f"Best DL model: {best_model_name}, R2: {best_model_metrics['R2']:.4f}, RMSE: {best_model_metrics['RMSE']:.4f}")
    print(f"Best DL model saved at: {model_file_path}")

    fig = model_instance.DL_model_comparison_scatter(X_train, X_test, Y_train, Y_test)
    fig_path = os.path.join(save_dir, "DL_model_comparison_scatter.png")
    fig.savefig(fig_path, dpi=300)
    print(f"DL model comparison figure saved at: {fig_path}")

    return fig, best_model_name, metrics, model_file_path

if __name__ == "__main__":
    file_path = r'best_features.xlsx'
    res_df = pd.read_excel(file_path, sheet_name='Sheet1')
    save_dir = r'D:\\Code_Store\\InversionSoftware\\S2_model_training\\Best_Model'
    DL_comparison_scatter_fig, best_model_name, metrics, model_file_path = DL_training_main(
        res_df, save_dir
    )
    print(metrics)
    DL_comparison_scatter_fig.show()
