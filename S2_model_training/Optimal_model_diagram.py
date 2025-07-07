"""
===============================================================
Module Name: Optimal_model_diagram

This script performs:
- Loading different types of regression models (Keras, Sklearn, TPOT, AutoGluon)
- Making unified predictions on training and test datasets
- Plotting scatter plots of true vs. predicted values
- Plotting comparison diagrams showing multiple models in one figure

Main Features:
- Model loading
- Universal prediction interface
- Visualization of results
- Saving model comparison plots
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import load_model
import seaborn as sns
from scipy.stats import linregress
import tensorflow as tf
from autogluon.tabular import TabularPredictor
from tpot import TPOTRegressor
import warnings

warnings.filterwarnings("ignore")

def load_model_from_path(model_path, model_type):
    """
    Load a model object based on its type and file path.

    Supported:
        - Keras models (.keras file)
        - Sklearn / TPOT models saved via joblib (.joblib file)
        - AutoGluon models (directory path)

    Parameters:
        model_path (str): Path to the model file
        model_type (str): Model type indicator to determine how to load

    Returns:
        Loaded model object
    """
    if model_path.endswith(".keras"):
        return load_model(model_path)
    elif model_path.endswith(".joblib"):
        return joblib.load(model_path)
    elif model_type == "AutoGluon":
        return TabularPredictor.load(model_path)
    elif model_type == "TPOT":
        return joblib.load(model_path)
    else:
        raise ValueError("Unsupported model file type. Please provide a '.keras' or '.joblib' file.")


def reshape_for_dl(X):
    """
    Convert input to a NumPy array and add a channel dimension for deep learning models.

    DL input shape:
        (samples, features, 1)

    Parameters:
        X (array-like or DataFrame): Input data

    Returns:
        numpy.ndarray: shape=(samples, features, 1)
    """
    if hasattr(X, 'values'):
        X = X.values
    return X[..., np.newaxis]


def safe_predict(model, X, model_type=None, feature_names=None, verbose=0):
    """
    Universal prediction interface compatible with various model types.

    Supported:
        - Keras models
        - AutoGluon models
        - TPOT models
        - Sklearn models

    Parameters:
        model: Loaded model object
        X: Input features (DataFrame, array, or tensor)
        model_type: Model type string
        feature_names: For arrays, AutoGluon requires feature names
        verbose: Whether to display progress bar (effective for Keras)

    Returns:
        np.ndarray: Predicted values (1D array)
    """
    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return model.predict(np.array(X), verbose=verbose).flatten()
    elif isinstance(model, TabularPredictor):
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                raise ValueError("[AutoGluon] Feature names are required for array input")
            X = pd.DataFrame(X, columns=feature_names)
        return model.predict(X).to_numpy().flatten()
    elif isinstance(model, TPOTRegressor):
        return model.predict(X).flatten()
    else:
        return model.predict(X.values if hasattr(X, 'values') else np.array(X)).flatten()


def plot_comparison(model, model_type, X_train, X_test, Y_train, Y_test,
                    title='Optimal Model Prediction Results', ax=None, feature_names=None):
    """
    Visualize the prediction performance of a model by plotting predicted vs. true values.

    - Separate scatter plots for training and test sets
    - Draw a reference line y = x
    - Draw fitted regression lines
    - Display RMSE and R² metrics on the plot

    Parameters:
        model: Loaded model object
        model_type (str): Model type string
        X_train, X_test: Feature training and test sets
        Y_train, Y_test: Target training and test sets
        title (str): Plot title
        ax (matplotlib.axes.Axes or None): If None, create a new figure
        feature_names (list or None): Feature names, required only for AutoGluon

    Returns:
        If ax is None:
            matplotlib.figure.Figure
        Otherwise:
            matplotlib.axes.Axes
    """
    dl_models = ['CNN', 'LSTM', 'GRU', 'BiLSTM', 'CNN_LSTM']
    if model_type in dl_models:
        X_train_in = reshape_for_dl(X_train)
        X_test_in = reshape_for_dl(X_test)
    else:
        X_train_in, X_test_in = X_train, X_test

    x_pred = safe_predict(model, X_train_in, model_type, feature_names)
    y_pred = safe_predict(model, X_test_in, model_type, feature_names)

    create_new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        create_new_fig = True

    sns.set_theme(style='whitegrid')
    ax.scatter(Y_test, y_pred, c='royalblue', edgecolors='k', marker='o', alpha=0.8, label='Test Data')
    ax.scatter(Y_train, x_pred, c='lightcoral', edgecolors='k', marker='^', alpha=0.8, label='Train Data')

    # Automatically set axis limits
    y_all = np.concatenate([Y_train, Y_test, x_pred, y_pred])
    y_min, y_max = y_all.min(), y_all.max()
    margin = (y_max - y_min) * 0.05

    ax.set_xlim(y_min - margin, y_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Plot y = x reference line
    ax.plot([y_min, y_max], [y_min, y_max], color='gray', linestyle='--', linewidth=1.5, label='y = x')

    # Plot fitted regression lines
    slope_test, intercept_test, _, _, _ = linregress(Y_test, y_pred)
    slope_train, intercept_train, _, _, _ = linregress(Y_train, x_pred)
    ax.plot([y_min, y_max], slope_test * np.array([y_min, y_max]) + intercept_test,
            color='green', linestyle='-', linewidth=2,
            label=f'Test Fit: y={slope_test:.2f}x+{intercept_test:.2f}')
    ax.plot([y_min, y_max], slope_train * np.array([y_min, y_max]) + intercept_train,
            color='black', linestyle='-', linewidth=2,
            label=f'Train Fit: y={slope_train:.2f}x+{intercept_train:.2f}')

    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title if title else model_type, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    ax.text(y_min + margin, y_max - margin * 3,
            f'RMSE: {np.sqrt(mean_squared_error(Y_test, y_pred)):.2f}\n$R^2$: {r2_score(Y_test, y_pred):.2f}',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    if create_new_fig:
        plt.tight_layout()
        plt.show()
        return plt.gcf()
    else:
        return ax


def plot_main(res_df, model_save_dir, model_type):
    """
    Main function: plots a comparison of predictions for multiple models on the test set.

    Process:
        - Load each model
        - Perform predictions on training and test sets
        - Plot scatter plots of true vs. predicted values
        - Arrange multiple subplots into one figure
        - Save the final comparison figure

    Parameters:
        res_df (pd.DataFrame):
            Dataset where:
                Column 1 is the target
                Columns from 2 onward are features
        model_save_dir (str):
            Directory where models are saved
        model_type (list[str]):
            List of model names to be plotted

    Returns:
        matplotlib.figure.Figure:
            The overall plotted figure object
    """
    # Split into training and test sets
    X = res_df.iloc[:, 1:]
    y = res_df.iloc[:, 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_models = len(model_type)
    cols = 3
    rows = (num_models + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axs = axs.flatten()

    for i, best_model in enumerate(model_type):
        # Automatically generate model path
        if best_model in ["AutoGluon", "AutoGluon_v1", "AutoGluon_best"]:
            model_path = os.path.join(model_save_dir, 'AutoGluon')
        elif best_model == "TPOT":
            model_path = os.path.join(model_save_dir, 'TPOT.pkl')
        elif best_model in ["SVR", "MLR", "RandomForest", "KNN", "GBDT", "XGBoost", "ELM", "BPNN", "PLSR"]:
            model_path = os.path.join(model_save_dir, f'{best_model}_best_ML_model.joblib')
        elif best_model in ["CNN", "LSTM", "GRU", "BiLSTM", "DNN", "CNN_LSTM", "MLP", "Autoencoder", "GNN (Mock)"]:
            model_path = os.path.join(model_save_dir, f'{best_model}_best_DL_model.keras')
        else:
            print(f"[Skipped] Model type {best_model} is not recognized.")
            continue

        try:
            model = load_model_from_path(model_path, best_model)
            plot_comparison(model, best_model, X_train, X_test, Y_train, Y_test, ax=axs[i])
            axs[i].set_title(f"Model: {best_model}")
        except Exception as e:
            axs[i].set_visible(False)
            print(f"[Error] Plotting failed for model {best_model}: {e}")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Performance of Optimal ML, DL, and AutoML Models on Full Dataset",
                 fontsize=14, fontweight='bold')
    save_path = os.path.join(model_save_dir, "best_model_comparison_results.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Optimal model comparison plot saved at: {save_path}")

    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    """
    Entry point of the script.

    Process:
        - Read data from Excel file
        - Call plot_main() to generate comparison plots for multiple models
        - Save the figure to the specified directory
    """
    file_path = r'best_features.xlsx'
    res_df = pd.read_excel(file_path, sheet_name='Sheet1')
    model_save_dir = r'D:\Code_Store\InversionSoftware\S2_model_training\Best_Model'
    model_type = ['GBDT', 'CNN', 'AutoGluon']
    plot_main(res_df, model_save_dir, model_type)
