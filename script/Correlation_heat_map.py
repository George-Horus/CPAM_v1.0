import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Pearson_correlation_lower_heatmap(df, target_col, corr_threshold=0.4):
    """
    Plot a lower triangle heatmap of Pearson correlation coefficients,
    keeping the diagonal (self-correlations), and removing grid lines.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and the target column.
        target_col (str): Name of the target column.
        corr_threshold (float): Minimum absolute correlation threshold
                                to consider features correlated with the target.

    Returns:
        correlations_dict (dict): Dictionary of correlations with the target.
        filtered_df (pd.DataFrame): DataFrame containing the target and selected features.
        fig (matplotlib.figure.Figure): The heatmap figure object.
    """
    # Compute correlation with the target column
    correlation_series = df.corr()[target_col].drop(target_col)
    filtered_corr = correlation_series[abs(correlation_series) > corr_threshold]

    if filtered_corr.empty:
        print(f"⚠️ No features found with correlation to target '{target_col}' exceeding threshold {corr_threshold}.")
        return {}, df[[target_col]], None

    selected_features = filtered_corr.index.tolist()
    correlations_dict = filtered_corr.to_dict()

    selected_cols = [target_col] + selected_features
    filtered_df = df[selected_cols]
    corr_matrix = filtered_df.corr()

    # Create a mask to hide the upper triangle (keep diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(selected_cols)), max(6, len(selected_cols))))
    sns.set(style="whitegrid", font_scale=0.9)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0,  # ✅ Remove grid lines
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax
    )
    ax.set_title(f"Lower Triangle Correlation Heatmap (|r| > {corr_threshold})", fontsize=14, fontweight='bold')

    return correlations_dict, filtered_df, fig

if __name__ == "__main__":
    target_value = 'N'
    save_dir = r'/'

    # Load the data
    df = pd.read_excel('original_data.xlsx')

    # Compute correlation heatmap
    corr_dict, df_filtered, fig = Pearson_correlation_lower_heatmap(df, target_value)

    if fig is not None:
        fig.savefig("correlation_heatmap_target_only.png", dpi=300, bbox_inches='tight')
        print("✅ Heatmap saved as correlation_heatmap_target_only.png")
    else:
        print("No heatmap generated because no features met the correlation threshold.")
