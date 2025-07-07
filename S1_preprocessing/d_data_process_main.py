"""
===============================================================
Script Name: d_data_process_main.py

Script Overview
---------------------------------------------------------------
This script is the data preprocessing module for remote sensing
inversion research or modeling software. Its core functions include:

1. **Batch processing of multiple remote sensing images (TIF files)**
    - Extract multispectral band values corresponding to ground
      sampling points or vector polygons from single-date or multi-date images.
    - Automatically calculate various vegetation indices.
    - Supports extraction of single pixel values or mean values
      in a surrounding region (moving window).

2. **Flexible adaptation to different experimental designs**
    - **One-to-one mode** (processing_tif1)
      Each TIF image corresponds to one sampling file (e.g. multi-temporal sampling analysis).
    - **Many-to-one mode** (processing_tif2)
      Multiple TIF images share the same sampling file, useful for
      yield inversion or time-series modeling.

3. **Feature Engineering**
    - Pearson correlation analysis to select bands or indices
      highly correlated with the target variable.
    - Feature selection using RFE (Recursive Feature Elimination)
      algorithm based on a Random Forest regressor.
    - Automatically generates various visualization charts
      suitable for scientific papers or reports.

4. **Automatic saving of results**
    - All intermediate results and final tables are automatically
      saved as Excel files.
    - Image results (e.g. correlation plots, RFE curves,
      feature importance plots) are saved in PNG format
      with high resolution (up to 300 dpi).

---------------------------------------------------------------
Input Data
---------------------------------------------------------------
- **List of TIF image file paths**
    Can be single-date or multi-date multi-band remote sensing images.

- **List of sampling point or vector polygon file paths**
    Supported formats: .csv, .xlsx, .txt, .shp
    Contains latitude/longitude coordinates and ground measurements
    (e.g. nitrogen content, water content, yield, etc.).

- **Band name list band_match**
    e.g. ['GREEN','RED','RedEdge','NIR'], used to map band indices
    to specific names for consistent index calculation.

- **Target variable column name target_value**
    For example 'WC' (Water Content), 'N' (Nitrogen content),
    or 'production' (Yield).

- **dataset_selection**
    Optional parameter for selecting a subset of images to process.

- **windowsize**
    Window size used when extracting pixel values
    (only applicable for point data).

---------------------------------------------------------------
Output Data
---------------------------------------------------------------
- Excel files:
    - all_indices.xlsx: all extracted band values and vegetation indices.
    - best_indices.xlsx: data table of optimal features after feature selection.

- PNG images:
    - Correlation_analysis.png
    - Feature_Selection_Optimization.png
    - Feature_Importance_Score.png
===============================================================
"""

from S1_preprocessing import b_Calculate_indices, c_Feature_filter
import os
import pandas as pd

def processing_tif1(tif_path_list, file_path_list, band_match, target_value, dataset_selection=None, windowsize=10):
    """
    Batch process multiple TIF image files, calculate vegetation indices,
    and merge the results into a single table. Used in scenarios where
    each image matches one sampling dataset, e.g. for inversion of
    nitrogen content, biomass, or water content.

    Parameters:
    ----------
    tif_path_list : list of str
        List of file paths for each input remote sensing image (TIF files).
    file_path_list : list of str
        List of file paths for data files (lat/lon + measurements)
        corresponding to each TIF image.
    band_match : list
        List defining the order of bands to prevent incorrect
        vegetation index calculations.
    target_value : str
        Column name of the measured value in the data file,
        to merge into the final DataFrame.
    dataset_selection : list[int] or None
        Optional: if provided, only processes the selected datasets.
    Returns:
    ----------
    final_df : pandas.DataFrame
        Final DataFrame merging results from all processed images.
    all_dfs : list of pandas.DataFrame
        List of DataFrames from each individual image, in order.
    """
    if dataset_selection is None:
        selected_tif_paths = tif_path_list
        selected_file_paths = file_path_list
    else:
        selected_tif_paths = [tif_path_list[i] for i in dataset_selection]
        selected_file_paths = [file_path_list[i] for i in dataset_selection]

    all_dfs = []
    for j in range(len(selected_tif_paths)):
        final_indice_df = b_Calculate_indices.indeces_main(
            selected_tif_paths[j],
            selected_file_paths[j],
            band_match,
            target_value,
            windowsize
        )
        all_dfs.append(final_indice_df.reset_index(drop=True))
    final_df = pd.concat(all_dfs, ignore_index=True)
    return final_df

def processing_tif2(tif_path_list, file_path, band_match, target_value, dataset_selection=None, windowsize=10):
    """
    Process multi-date remote sensing TIF images, extract vegetation indices,
    and merge them into a training dataset. Used when multiple remote sensing
    images correspond to the same sampling point data, e.g. for yield inversion.

    Function:
    ----------
    - Extract vegetation indices from multiple TIF images using the same
      ground sampling data.
    - Each image's features are suffixed with the period number
      (e.g. NDVI_1, NDVI_2).
    - Retain only one target_value column across all images
      (the first occurrence) and move it to the first column.
    - Outputs a dataset ready for modeling.

    Parameters:
    ----------
    tif_path_list : list of str
        List of file paths for multi-date remote sensing images.
    file_path : list of str
        List containing a single file path to ground sampling data
        (lat/lon + measured values like yield).
    band_match : list of str
        Band order matching list, e.g. ['BLUE', 'GREEN', 'NIR', 'RED', 'RedEdge'].
    target_value : str
        Name of the target variable in the measured data
        (e.g. 'production' for yield).
    dataset_selection : list[int] or None
        Optional: list of indices to select specific images for processing.

    Returns:
    ----------
    final_df : pd.DataFrame
        DataFrame horizontally merging all images,
        with the target variable as the first column.
    """

    if dataset_selection is None:
        selected_tif_paths = tif_path_list
    else:
        selected_tif_paths = [tif_path_list[i] for i in dataset_selection]

    all_dfs = []

    for i, tif_path in enumerate(selected_tif_paths):
        df = b_Calculate_indices.indeces_main(
            tif_path,
            file_path[0],  # All images use the same ground sampling data
            band_match,
            target_value,
            windowsize
        ).reset_index(drop=True)

        # Add suffix _1, _2, _3, etc. to indicate which period each column belongs to
        df = df.rename(columns={col: f"{col}_{i + 1}" for col in df.columns})
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, axis=1)

    target_cols = [col for col in final_df.columns if col.startswith(target_value + "_")]
    keep_target_col = f"{target_value}_1"

    for col in target_cols:
        if col != keep_target_col:
            final_df.drop(columns=col, inplace=True)

    if keep_target_col in final_df.columns:
        final_df.rename(columns={keep_target_col: target_value}, inplace=True)

    if target_value in final_df.columns:
        cols = list(final_df.columns)
        cols.insert(0, cols.pop(cols.index(target_value)))
        final_df = final_df[cols]

    return final_df

def data_process_main(tif_path_list, file_path_list, band_match, target_value, save_dir, dataset_selection, windowsize):
    """
    Main function for the data preprocessing module. Handles
    reading data, extracting sampling point features, and
    performing feature engineering.
    """

    save_path = os.path.join(save_dir, 'Feature_Selection')
    os.makedirs(save_path, exist_ok=True)

    if len(tif_path_list) != len(file_path_list) and len(file_path_list) == 1:
        # Multiple images with a single table of ground data (e.g. yield estimation)
        final_df = processing_tif2(
            tif_path_list, file_path_list, band_match,
            target_value, dataset_selection, windowsize
        )
    else:
        final_df = processing_tif1(
            tif_path_list, file_path_list, band_match,
            target_value, dataset_selection, windowsize
        )

    excel_save_path = os.path.join(save_path, 'all_indices.xlsx')
    final_df.to_excel(excel_save_path, index=False)
    print(f"\n✅ all_indices data has been saved to Excel file: {save_path}")

    correlations_dict, filtered_features_df, feature_num1, fig = c_Feature_filter.Pearson_correlation_analysis(
        final_df, target_value
    )

    sorted_features, best_features_df, feature_num2, fig1, fig2 = c_Feature_filter.RFE_select_optimal_features(
        filtered_features_df, target_value
    )

    excel_save_path = os.path.join(save_path, 'best_indices.xlsx')
    best_features_df.to_excel(excel_save_path, index=False)
    print(f"\n✅ best_indices data has been saved to Excel file: {save_path}")

    fig_path = os.path.join(save_path, "Correlation_analysis.png")
    fig.savefig(fig_path, dpi=300)
    print(f"[Saved] Correlation analysis plot: {fig_path}")

    fig1_path = os.path.join(save_path, "Feature_Selection_Optimization.png")
    fig1.savefig(fig1_path, dpi=300)
    print(f"[Saved] Feature selection optimization plot: {fig1_path}")

    fig2_path = os.path.join(save_path, "Feature_Importance_Score.png")
    fig2.savefig(fig2_path, dpi=300)
    print(f"[Saved] Feature importance score plot: {fig2_path}")

    figures = {
        'fig': fig,
        'fig1': fig1,
        'fig2': fig2
    }
    return best_features_df, sorted_features, figures, save_path

# Example usage
if __name__ == "__main__":

    # Parameter settings
    tif_path = [r"240307_multi_30m.tif"]  # Path to TIFF image
    file_path = [r"ROI.shp"]  # File containing WGS84 lat/lon coordinates
    band_match = ['GREEN', 'RED', 'RedEdge', 'NIR']  # Meaning of each band
    target_value = 'WC'
    save_dir = r'D:result'
    dataset_selection = [0, 1, 2, 3, 4]
    windowsize = 10

    # Run the main data processing function
    best_features_df, sorted_features, figures, save_path = data_process_main(
        tif_path, file_path, band_match,
        target_value, save_dir,
        dataset_selection, windowsize
    )
    print(sorted_features)
    figures['fig'].show()
    figures['fig1'].show()
    figures['fig2'].show()
