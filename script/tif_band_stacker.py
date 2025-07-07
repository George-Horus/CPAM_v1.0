"""
Module Name: tif_band_stacker

This script is used to stack multiple single-band GeoTIFF files from the same
folder into a single multi-band GeoTIFF file.

Useful for:
- Multiple vegetation index layers
- Single-band images from different times
- Any remote sensing layers that need band merging

Main Features:
1. Read all .tif files in a given input folder.
2. Merge all single-band images into a multi-band array.
3. Save as a new multi-band GeoTIFF file while preserving georeferencing info.

Inputs:
- Input folder path
- Output file path

Outputs:
- Stacked multi-band GeoTIFF file
"""

import rasterio
import numpy as np
import glob
import os

def stack_tifs(input_folder, output_path):
    """
    Stack all TIF files in the specified folder into a multi-band TIF.

    Parameters:
        input_folder (str): Directory containing multiple single-band TIF files.
        output_path (str): Path to save the resulting multi-band TIF file.
    """

    # Get all TIF file paths
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))

    if not tif_files:
        raise FileNotFoundError(f"❌ No TIF files found in input folder: {input_folder}")

    print(f"Found {len(tif_files)} TIF files. Starting band stacking...")

    # List to store all bands
    band_data = []

    # Use the first image as the reference for metadata
    with rasterio.open(tif_files[0]) as src_ref:
        meta = src_ref.meta.copy()

    for tif in tif_files:
        with rasterio.open(tif) as src:
            data = src.read(1)  # Read only the first band
            band_data.append(data)
            print(f"Loaded band: {os.path.basename(tif)}")

    # Stack as (bands, height, width)
    stacked_array = np.stack(band_data)

    # Update meta info for multi-band
    meta.update(count=len(band_data))

    # Save the stacked image
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(stacked_array)

    print(f"✅ Stacking completed. Output saved to: {output_path}")


if __name__ == "__main__":
    # ===== Example usage =====
    input_folder = r"D:\Zyh\20250421indices"  # Replace with your own path
    output_path = r"D:\Zyh\20250421comp.tif"

    stack_tifs(input_folder, output_path)
