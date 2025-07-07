"""
Module Name: tif_analysis_tools

This module provides automated analysis tools for remote sensing GeoTIFF images.

Main Features:
1. Calculate GSD (Ground Sampling Distance) for each GeoTIFF image
    - Automatically detect image coordinate system
    - If the image is in a geographic coordinate system (units: degrees), automatically reproject to a metric coordinate system (default EPSG:3857) before calculating GSD
    - Outputs image dimensions, band count, data type, NoData value, CRS information, etc.
    - Supports batch scanning of an entire folder and exporting results to Excel

2. Plot pixel value histograms for GeoTIFF images
    - Supports single image or batch plotting for a folder
    - Supports saving individual histogram images
    - Supports combined overview plots

Inputs:
- Single GeoTIFF file path
- Or folder path containing multiple GeoTIFF files

Outputs:
- GSD information printed in console for each image
- tif_GSD_info_auto_projection.xlsx file
- Pixel value histogram PNG files
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from glob import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import pandas as pd

# Set Chinese font for plotting if needed; remove this for purely English environments
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def plot_tif_histograms(input_path):
    """
    Plot histograms of pixel values (in the range 0–1) for one or more .tif files.

    If the input is a folder, automatically loads all .tif files and plots
    subplots, saving the images as PNG.

    Parameters:
        input_path (str): Single .tif file path or folder path containing .tif files.
    """
    if os.path.isfile(input_path) and input_path.lower().endswith('.tif'):
        tif_files = [input_path]
        output_dir = os.path.join(os.path.dirname(input_path), "histograms")
    elif os.path.isdir(input_path):
        tif_files = sorted(glob(os.path.join(input_path, '*.tif')))
        output_dir = os.path.join(input_path, "histograms")
    else:
        print("❌ Invalid input path. Please provide a valid .tif file or folder path.")
        return

    if not tif_files:
        print("⚠️ No .tif files found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    num_files = len(tif_files)
    cols = 3
    rows = int(np.ceil(num_files / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_files > 1 else [axes]

    for idx, tif_file in enumerate(tif_files):
        with rasterio.open(tif_file) as src:
            bands = src.read()
            pixels = bands.flatten()
            valid = pixels[(pixels >= 0) & (pixels <= 1)]

            print(f"\n📂 File: {os.path.basename(tif_file)}")
            print(f"Valid pixels: {len(valid)}")
            print(f"Mean: {np.mean(valid):.4f}, Median: {np.median(valid):.4f}, StdDev: {np.std(valid):.4f}")

            # Plot single histogram and save
            fig_single, ax = plt.subplots(figsize=(8, 5))
            ax.hist(valid, bins=50, color='gray', edgecolor='black', alpha=0.7)
            ax.set_title(f"{os.path.basename(tif_file)}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            filename = os.path.splitext(os.path.basename(tif_file))[0] + "_hist.png"
            save_path = os.path.join(output_dir, filename)
            fig_single.tight_layout()
            fig_single.savefig(save_path, dpi=300)
            plt.close(fig_single)

            # Also plot into combined figure
            ax_all = axes[idx]
            ax_all.hist(valid, bins=50, color='gray', edgecolor='black', alpha=0.7)
            ax_all.set_title(f"{os.path.basename(tif_file)}")
            ax_all.set_xlabel("Pixel Value")
            ax_all.set_ylabel("Frequency")
            ax_all.grid(True)

    for j in range(num_files, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    print(f"\n✅ All histograms have been saved in folder: {output_dir}")

def calculate_gsd_with_projection(tif_path):
    """
    Calculate GSD (Ground Sampling Distance) for a single GeoTIFF.

    Automatically reprojects if CRS is geographic (degrees).

    Parameters:
        tif_path (str): Path to the GeoTIFF file.

    Returns:
        dict: GSD information for the image.
    """
    with rasterio.open(tif_path) as src:
        crs = src.crs
        xres = src.transform[0]
        yres = -src.transform[4]

        if crs.is_geographic:
            print(f"🌍 Reprojecting for metric GSD: {os.path.basename(tif_path)}")
            dst_crs = "EPSG:3857"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            xres_meters = transform[0]
            yres_meters = -transform[4]
        else:
            xres_meters = xres
            yres_meters = yres

        return {
            "Filename": os.path.basename(tif_path),
            "Width": src.width,
            "Height": src.height,
            "Band Count": src.count,
            "Data Type": src.dtypes[0],
            "NoData Value": src.nodata,
            "Original CRS": crs.to_string(),
            "CRS Units": "Degrees" if crs.is_geographic else "Meters",
            "GSD_X (m)": round(xres_meters, 4),
            "GSD_Y (m)": round(yres_meters, 4),
            "GSD (Average)": round((xres_meters + yres_meters) / 2, 4)
        }

def batch_scan_tif_with_gsd(directory, save_excel=True):
    """
    Scan all TIF files in a folder and calculate GSD info.

    Parameters:
        directory (str): Path to the folder containing .tif files.
        save_excel (bool): Whether to save results to Excel.
    """
    tif_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".tif")
    ]
    all_info = []

    for tif_path in tif_files:
        try:
            info = calculate_gsd_with_projection(tif_path)
            all_info.append(info)
        except Exception as e:
            print(f"❌ Failed to process {tif_path}: {e}")

    df = pd.DataFrame(all_info)
    print("\n📋 All image information:\n")
    print(df)

    if save_excel:
        output_path = os.path.join(directory, "tif_GSD_info_auto_projection.xlsx")
        df.to_excel(output_path, index=False)
        print(f"\n✅ Info saved to: {output_path}")


if __name__ == "__main__":
    input_path = r"D:\Zyh"
    batch_scan_tif_with_gsd(input_path)
    plot_tif_histograms(input_path)
