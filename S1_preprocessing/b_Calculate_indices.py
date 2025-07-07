"""
===============================================================
Script Name: b_Calculate_indices.py

Script Function:
-----------------------------------------------
This script extracts band values for sample points from
multi-band remote sensing images and computes various vegetation
indices. It then combines the target values (e.g. nitrogen content, N)
associated with the sample points with the extracted band values
and vegetation indices to produce a training dataset suitable
for modeling. It supports multiple sample point file formats,
including csv, xlsx, txt, and shp.

Main functions include:
    - Extracting band values from remote sensing images
    - Replacing band column names with specified names
    - Computing various vegetation indices using the VMID index library
    - Merging target values of sampling points with calculated indices
    - Exporting a complete training dataset table

Input Parameters:
-----------------------------------------------
- tif_path (str)
    Path to multi-band remote sensing imagery

- file_path (str)
    Path to the sampling point data (supports csv, xlsx, txt, shp)

- band_match (list of str)
    A list specifying the names corresponding to each band,
    e.g. ["BLUE", "GREEN", "RED", "RedEdge", "NIR"]

- target_value (str)
    Column name of the target variable in the sampling point file
    for modeling (e.g. 'N')

- windowsize (int, optional)
    Window size used for extracting pixel values from imagery. Default is 10.

Output:
-----------------------------------------------
- pandas.DataFrame
    A table containing the merged target values, band values,
    and vegetation indices, suitable for further modeling.

Usage Example:
-----------------------------------------------
Please refer to the example code at the end of the script under __main__.
===============================================================
"""

from S1_preprocessing import a_Extract_band_value, VMID_1
import os
import pandas as pd
import geopandas as gpd

def replace_columns(df, band_match):
    """
    Rename bands in the input TIFF to user-specified band names, e.g. band1 -> RED.
    It searches for columns in the DataFrame starting with 'band' and ending
    with digits, and replaces them sequentially with names from band_match.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        band_match (list): List of new band names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    band_cols = sorted(
        [col for col in df.columns if col.startswith('band') and col[4:].isdigit()],
        key=lambda x: int(x[4:])
    )
    rename_dict = {old: new for old, new in zip(band_cols, band_match)}
    return df.rename(columns=rename_dict)

def Calculate_indices(df):
    """
    Compute vegetation indices using the VMID vegetation index library.

    Parameters:
        df (pd.DataFrame): DataFrame containing band values.

    Returns:
        pd.DataFrame: DataFrame with all computed vegetation indices appended.
    """
    red = df["RED"] if "RED" in df.columns else None
    green = df["GREEN"] if "GREEN" in df.columns else None
    blue = df["BLUE"] if "BLUE" in df.columns else None
    red_edge = df["RedEdge"] if "RedEdge" in df.columns else None
    nir = df["NIR"] if "NIR" in df.columns else None

    vminstance = VMID_1.VMID(red, green, blue, red_edge, nir)

    df1 = vminstance.export_all()
    indice_df = pd.concat([df, df1], axis=1)

    return indice_df

def concat_final_dataframe(df, file_path, target_value):
    """
    Merge the target variable values with the computed indices to
    form a training dataset.

    Removes 'lon', 'lat', and 'index' columns from the input DataFrame if present.
    Reads the target variable from the provided file (csv/xlsx/txt/shp) and
    places it as the first column in the merged DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame with band and index data.
        file_path (str): Path to the file containing sample point data.
        target_value (str): The column name of the target variable.

    Returns:
        pd.DataFrame: The merged DataFrame ready for modeling.
    """
    df = df.drop(columns=[col for col in ['lon', 'lat', 'index'] if col in df.columns])

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        file_df = pd.read_csv(file_path)
    elif ext == '.xlsx':
        file_df = pd.read_excel(file_path)
    elif ext == '.txt':
        file_df = pd.read_csv(file_path, delimiter='\t')
    elif ext == '.shp':
        file_df = gpd.read_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if target_value not in file_df.columns:
        raise ValueError(f"Column {target_value} not found in the file.")

    value_col = file_df[[target_value]].reset_index(drop=True)
    df = df.reset_index(drop=True)
    final_indice_df = pd.concat([value_col, df], axis=1)

    return final_indice_df

def indeces_main(tif_path, file_path, band_match, target_value, windowsize=10):
    """
    Main function for calculating vegetation indices for sampling points.

    Given a TIFF image, sample point coordinates, and corresponding
    target values, it outputs a dataset ready for modeling.

    Parameters:
        tif_path (str): Path to the multi-band remote sensing image.
        file_path (str): Path to the sampling point data.
        band_match (list): List of band names to assign.
        target_value (str): Name of the target variable column.
        windowsize (int): Window size for extracting band values.

    Returns:
        pd.DataFrame: The merged dataset ready for modeling.
    """
    df = a_Extract_band_value.Extract_band_value_main(tif_path, file_path, windowsize)
    df = replace_columns(df, band_match)
    indice_df = Calculate_indices(df)
    final_indice_df = concat_final_dataframe(indice_df, file_path, target_value)
    return final_indice_df


if __name__ == "__main__":
    # Example usage

    tif_path = r"20250421_wheat.tif"
    file_path = r"20250421_sample.xlsx"
    band_match = ["BLUE", "GREEN", "RED", "RedEdge", "NIR"]
    target_value = "N"

    final_df = indeces_main(tif_path, file_path, band_match, target_value)
    final_df.to_excel("20250421_indices.xlsx", index=False)

    print("✅ Index calculation and data merging completed. Results have been saved.")
