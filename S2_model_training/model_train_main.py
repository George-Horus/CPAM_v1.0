"""
===============================================================
Module Name: model_train_main

This script performs:
1. Data preprocessing (normalization).
2. Model training:
    - Traditional machine learning models (ML)
    - Deep learning models (DL)
    - AutoML models
3. Outputs:
    - Performance metrics tables for different models
    - Various plots (model results, model comparisons, etc.)
    - Paths to saved model files
    - Directory where all outputs are stored

Main Features:
- Model training and comparison
- Saving model performance metrics to Excel
- Generating bar charts for performance visualization
- Generating comparative plots across different models
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from S2_model_training import ML_model, AutoML_model, Optimal_model_diagram, DL_model
import warnings
warnings.filterwarnings("ignore")

def convert_model_metrics_to_dataframe(ml_dict, dl_dict, automl_dict):
    """
    Converts evaluation results from ML, DL, and AutoML models
    into a unified DataFrame format.

    Parameters:
        ml_dict (dict):
            Dictionary of performance metrics from traditional ML models.
            Format:
                {
                    'ModelName1': {'R2': float, ... other metrics},
                    'ModelName2': {...},
                    ...
                }
        dl_dict (dict):
            Dictionary of performance metrics from DL models, same format.
        automl_dict (dict):
            Dictionary of performance metrics from AutoML models, same format.

    Returns:
        pd.DataFrame:
            A DataFrame with columns:
                - Model: model name
                - Type: model type (ML, DL, AutoML)
                - R2: R² score
                - Other metrics (e.g. RMSE, MAE, etc. depending on dictionary content)
            Sorted in descending order by R².
    """
    all_data = []

    # Add ML models
    for model_name, metrics in ml_dict.items():
        row = {'Model': model_name, 'Type': 'ML'}
        row.update(metrics)
        all_data.append(row)

    # Add DL models
    for model_name, metrics in dl_dict.items():
        row = {'Model': model_name, 'Type': 'DL'}
        row.update(metrics)
        all_data.append(row)

    # Add AutoML models
    for model_name, metrics in automl_dict.items():
        row = {'Model': model_name, 'Type': 'AutoML'}
        row.update(metrics)
        all_data.append(row)

    df = pd.DataFrame(all_data)

    # Sort by R² in descending order
    df = df.sort_values(by='R2', ascending=False).reset_index(drop=True)

    return df


def visualize_model_performance(df, save_dir):
    """
    Plots a bar chart comparing R² scores for all models and saves
    the figure to the specified directory.

    Parameters:
        df (pd.DataFrame):
            DataFrame containing model performance, must include:
                - 'Model': model names
                - 'R2': R² scores
                - 'Type': model type (ML, DL, AutoML)
        save_dir (str):
            Directory path to save the plot.

    Returns:
        matplotlib.figure.Figure:
            The generated bar chart Figure object.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2', hue='Type', data=df, palette='Set2')
    plt.ylim(0, 1)
    plt.title("R² Comparison of Models", fontsize=8, fontweight='bold')
    plt.ylabel("R² Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    # Save bar chart
    barplot_path = os.path.join(save_dir, "Model_R2_Barplot.png")
    plt.savefig(barplot_path, dpi=300)
    print(f"[Saved] R² comparison bar plot: {barplot_path}")
    return plt.gcf()


def Training_main(res_df, save_dir):
    """
    Executes the entire model training pipeline:
        - Data normalization (only on columns with mean > 1)
        - Trains ML, DL, and AutoML models
        - Aggregates performance metrics
        - Generates various plots
        - Saves results locally

    Parameters:
        res_df (pd.DataFrame):
            Raw input data table.
            Requirements:
                The first column is the target variable.
                From the second column onward are feature columns.
        save_dir (str):
            Root directory to save model files, plots, and performance tables.

    Returns:
        tuple:
            - model_performance_df (pd.DataFrame):
                Table summarizing model performance metrics.
            - figures (dict):
                Dictionary of figure objects:
                    {
                        'ML': matplotlib Figure,
                        'DL': matplotlib Figure,
                        'AutoML': matplotlib Figure,
                        'BestModel': matplotlib Figure,
                        'Barplot': matplotlib Figure
                    }
            - model_path (list):
                File paths of the best models:
                    [ML_model_file_path, DL_model_file_path, AutoML_model_file_path]
            - save_path (str):
                Subdirectory path where all results are saved (Best_Model).
    """
    # Set save directory
    save_path = os.path.join(save_dir, 'Best_Model')
    os.makedirs(save_path, exist_ok=True)

    # Feature normalization (only for columns with mean > 1)
    df_normalized = res_df.copy()
    features = df_normalized.columns[1:]  # first column is the target
    mean_values = df_normalized[features].mean()
    cols_to_normalize = mean_values[mean_values > 1].index.tolist()
    if cols_to_normalize:
        scaler = MinMaxScaler()
        df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

    # Train all models
    ML_fig, best_ML_model_name, ML_metrics, ML_model_file_path = ML_model.ML_training_main(res_df, save_path)
    DL_fig, best_DL_model_name, DL_metrics, DL_model_file_path = DL_model.DL_training_main(res_df, save_path)
    AutoML_fig, best_AutoML_model_name, AutoML_metrics, AutoML_model_file_path = AutoML_model.AutoML_training_main(res_df, save_path)

    # Aggregate model performance
    model_performance_df = convert_model_metrics_to_dataframe(ML_metrics, DL_metrics, AutoML_metrics)
    # print(model_performance_df)

    # Save performance table to Excel
    metrics_excel_path = os.path.join(save_path, "Model_Performance_Summary.xlsx")
    model_performance_df.to_excel(metrics_excel_path, index=False)
    print(f"[Saved] Model performance summary: {metrics_excel_path}")

    # Comparison figure of best models
    model_type = [best_ML_model_name, best_DL_model_name, best_AutoML_model_name]
    model_path = [ML_model_file_path, DL_model_file_path, AutoML_model_file_path]
    best_model_compfig = Optimal_model_diagram.plot_main(res_df, save_path, model_type)

    # Bar chart of performance
    Model_R2_Barplot_fig = visualize_model_performance(model_performance_df, save_path)

    # Figures dictionary for easy access
    figures = {
        'ML': ML_fig,
        'DL': DL_fig,
        'AutoML': AutoML_fig,
        'BestModel': best_model_compfig,
        'Barplot': Model_R2_Barplot_fig
    }

    return model_performance_df, figures, model_path, save_path


if __name__ == "__main__":
    """
    Entry point of the program.

    Features:
        - Loads input data from an Excel file.
        - Calls Training_main() to run the entire training pipeline.
        - Prints the model performance table and paths of saved models.
        - Displays all plots.
    """
    file_path = r'D:\Code_Store\InversionSoftware\S2_model_training\best_features.xlsx'
    res_df = pd.read_excel(file_path, sheet_name='Sheet1')
    save_dir = r'D:\Code_Store\InversionSoftware\S2_model_training'
    model_performance_df, figures, model_path, save_path = Training_main(res_df, save_dir)
    print(model_performance_df)
    print(model_path)
    figures['ML'].show()
    figures['DL'].show()
    figures['AutoML'].show()
    figures['BestModel'].show()
    figures['Barplot'].show()
