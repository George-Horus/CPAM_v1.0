"""
===============================================================
Script Name: c_Feature_filter.py

Script Function:
-----------------------------------------------
This script implements a feature engineering workflow based on
remote sensing bands and vegetation indices:
    1. Perform Pearson correlation analysis
        - Calculate correlation coefficients between all features and the target variable
        - Visualize significantly correlated features
    2. Feature selection using RFE (Recursive Feature Elimination)
        - Based on a Random Forest model
        - Search for the optimal feature subset
        - Plot optimization curves (MSE & R² vs. number of features)
        - Plot feature importance for the optimal subset

Input:
-----------------------------------------------
- Excel file containing the target variable and all band/index features

Output:
-----------------------------------------------
- Excel file: table of filtered optimal features
- PNG images:
    - Bar chart showing correlations between features and the target variable
    - Optimization curve for feature selection
    - Bar chart of feature importance for the optimal subset
===============================================================
"""
import os
import numpy as np
from S1_preprocessing import b_Calculate_indices
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def Pearson_correlation_analysis(df, target_col):
    """
    Perform Pearson correlation analysis to compute the correlations
    between all vegetation index features and the target column,
    and visualize the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing the target column and feature columns.
        target_col (str): Name of the target column for correlation analysis.

    Returns:
        correlations_dict (dict): Dictionary of features with absolute
                                  correlation > 0.2 and their correlation coefficients.
        filtered_features_df (pd.DataFrame): DataFrame containing only
                                             features with absolute correlation > 0.2.
        feature_num (int): Total number of selected features.
        plt.gcf(): Matplotlib Figure object of the bar chart.

    Visualization:
        Generates a bar chart showing features with absolute
        correlation > 0.2 and their correlation coefficients.
        The chart is styled for scientific publication,
        with clear gridlines, a baseline, and color gradients
        for distinction.

    Usage Example:
        correlations_dict, filtered_features_df, feature_num = Pearson_correlation_analysis(df, 'water_content')
    """

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Extract correlation with the target column and sort
    target_correlation = correlation_matrix[target_col].drop(target_col)
    sorted_correlation = target_correlation.sort_values(ascending=False)

    # Filter features with |r| > 0.2
    filtered_correlation = sorted_correlation[abs(sorted_correlation) > 0.2]
    correlations_dict = filtered_correlation.to_dict()

    # Keep only the selected columns in the input DataFrame
    col_list = filtered_correlation.index.values
    filtered_cols = [col for col in df.columns if col in col_list]

    # Concatenate the target column with the filtered feature columns
    filtered_features_df = pd.concat([df[target_col], df[filtered_cols]], axis=1)

    feature_num = len(df.columns) - 1

    # Visualization in scientific style
    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    ax = sns.barplot(
        x=filtered_correlation.index,
        y=filtered_correlation.values,
        hue=filtered_correlation.index,
        palette=sns.color_palette("viridis", len(filtered_correlation)),
        legend=False
    )

    ax.set_title("Correlation analysis", fontsize=14, fontweight='bold')
    ax.set_xlabel("Features", fontsize=10)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)

    ax.set_xticks(range(len(filtered_correlation)))
    ax.set_xticklabels(filtered_correlation.index, rotation=90, ha='right', fontsize=6)

    ax.axhline(y=0, color='black', linewidth=1.2, linestyle="--")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    return correlations_dict, filtered_features_df, feature_num, plt.gcf()

def RFE_select_optimal_features(df, target_col):
    """
    Use RFE (Recursive Feature Elimination) to select the optimal feature subset,
    and plot optimization curves and feature importance charts.

    RFE Working Principle:
    1. Fit a base estimator model (here, a Random Forest).
    2. Calculate feature importances via coef_ or feature_importances_.
    3. Remove the least important features by step size.
    Repeat steps 1-3 until reaching the desired number of features.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and target values.
        target_col (str): Name of the target column.

    Returns:
        sorted_features (list): Names of the selected optimal features, sorted by importance.
        best_features_df (pd.DataFrame): DataFrame containing the optimal feature subset and the target column.
        feature_num (int): Number of final selected features.
        fig1 (matplotlib Figure): Optimization curve plot.
        fig2 (matplotlib Figure): Feature importance bar chart.
    """

    # Extract target and features
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Random Forest
    rf = RandomForestRegressor(random_state=42)

    # RFE feature selection
    selector = RFE(rf, n_features_to_select=1, step=3)
    selector.fit(X_train, y_train)

    # Get feature ranking
    ranking = selector.ranking_
    ranking_order = np.argsort(ranking)

    # Compute performance metrics for incremental feature subsets
    mse_scores = []
    r2_scores = []
    selected_features = []

    for n_features in range(1, X_train.shape[1] + 1):
        selected_cols = X_train.columns[ranking_order[:n_features]]
        X_train_selected = X_train[selected_cols]
        X_test_selected = X_test[selected_cols]

        rf.fit(X_train_selected, y_train)
        y_pred = rf.predict(X_test_selected)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        selected_features.append(list(selected_cols))

    optimal_features = np.argmin(mse_scores) + 1
    best_feature_set = selected_features[optimal_features - 1]
    feature_num = len(best_feature_set)

    rf.fit(X_train[best_feature_set], y_train)
    feature_importances = rf.feature_importances_

    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = [best_feature_set[i] for i in sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    filtered_cols = [col for col in df.columns if col in best_feature_set]
    best_features_df = pd.concat([df[target_col], df[filtered_cols]], axis=1)

    # Plot optimization curves
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("Mean Squared Error (MSE)", color='b')
    ax1.plot(range(1, X_train.shape[1] + 1), mse_scores, marker='o', color='b', label="MSE")
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel("R² Score", color='g')
    ax2.plot(range(1, X_train.shape[1] + 1), r2_scores, marker='s', color='g', label="R²")
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title("Feature Selection Optimization: MSE & R² vs. Feature Count")
    ax1.grid(True)
    fig1.tight_layout()

    # Plot feature importance bar chart
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(sorted_features[::-1], sorted_importances[::-1], color='skyblue')
    ax3.set_xlabel("Feature Importance Score")
    ax3.set_ylabel("Feature Name")
    ax3.set_title("Feature Importance of Optimal Feature Subset")
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    fig2.tight_layout()

    return sorted_features, best_features_df, feature_num, fig1, fig2

if __name__ == "__main__":

    tif_path = "data/20240525comp.tif"  # Path to TIFF imagery
    file_path = "data/20240525xy+N.xlsx"  # File containing WGS84 lat/lon coordinates
    band_match = ['BLUE', 'GREEN', 'NIR', 'RED', 'RedEdge']  # Meaning of each band
    target_value = 'N'
    save_dir = r'D:\Code_Store\InversionSoftware\S1_preprocessing'

    # Compute all vegetation indices
    final_indice_df = b_Calculate_indices.indeces_main(
        tif_path, file_path, band_match, target_value
    )

    # Perform correlation analysis
    correlations_dict, filtered_features_df, feature_num, fig = Pearson_correlation_analysis(
        final_indice_df, target_value
    )
    print(f"{correlations_dict}\n, {filtered_features_df}\n, {feature_num}\n")

    # Perform feature selection with RFE
    sorted_features, best_features_df, feature_num, fig1, fig2 = RFE_select_optimal_features(
        filtered_features_df, target_value
    )
    print(f"{sorted_features}\n, {best_features_df}\n, {feature_num}\n")

    # Save the final DataFrame to Excel
    best_features_df.to_excel('best_features.xlsx', index=False)

    # Save plots
    fig.show()
    fig_path = os.path.join(save_dir, "Correlation_analysis.png")
    fig.savefig(fig_path, dpi=300)
    print(f"[Saved] Correlation analysis plot: {fig_path}")

    fig1.show()
    fig1_path = os.path.join(save_dir, "Feature_Selection_Optimization.png")
    fig1.savefig(fig1_path, dpi=300)
    print(f"[Saved] Feature selection optimization plot: {fig1_path}")

    fig2.show()
    fig2_path = os.path.join(save_dir, "Feature_Importance_Score.png")
    fig2.savefig(fig2_path, dpi=300)
    print(f"[Saved] Feature importance score plot: {fig2_path}")
