"""
Module name: visualize_tif

This module is used for visualizing remote sensing prediction results (GeoTIFF). Core functionalities include:
    - Visualizing GeoTIFF data as color PNG/JPG images, with optional automatic outlier clipping.
    - Flexible colormap selection.
    - Plotting histograms of GeoTIFF pixel values to show data distributions before and after clipping.

Main features:
1. `visualize_tif`
    - Reads data from a GeoTIFF file.
    - Maps data values to color using a user-specified colormap.
    - Performs percentile-based clipping to reduce the influence of extreme values.
    - Saves output as PNG or JPG.
    - Returns a matplotlib Figure object for further processing or display.

2. `plot_percentile_clipping_hist`
    - Reads pixel data from a GeoTIFF file.
    - Plots histograms of pixel values before and after clipping, for analyzing data distribution and the effect of outlier removal.

Inputs:
- GeoTIFF file path
- Output image save path (PNG/JPG)
- Model type (used in automatic file naming)
- Colormap name (Matplotlib colormap)
- Whether to display the colorbar
- Clipping percentile

Outputs:
- Color PNG/JPG image file
- matplotlib Figure object (for GUI or further display)
- Histogram of pixel value distributions
"""

import matplotlib as mpl
from PIL import Image
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_tif(tif_path, output_img_path, model_type, colormap, show_colorbar=True, clip_percentile=5):
    """
    Visualizes a GeoTIFF image as PNG or JPG, with automatic color mapping and outlier clipping.

    Parameters
    ----------
    tif_path : str
        Input GeoTIFF file path.
    output_img_path : str or None
        Output image path (e.g. .png or .jpg) or folder path. If None or a folder path, the output filename is generated automatically.
    model_type : str
        Name for model type (used for automatic file naming).
    colormap : str
        Name of Matplotlib colormap, e.g. 'RdYlGn'.
    show_colorbar : bool
        Whether to display the colorbar.
    clip_percentile : float
        Outlier clipping percentile. E.g. 5 means clipping the lowest and highest 5%.

    Returns
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object for the visualization result.
    """

    # Read TIFF data
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)

    # Remove invalid values
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("❌ No valid data to visualize!")

    # Compute clipping bounds
    lower = np.percentile(valid, clip_percentile)
    upper = np.percentile(valid, 100 - clip_percentile)
    data_clipped = np.clip(data, lower, upper)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=lower, vmax=upper)
    colored = cmap(norm(data_clipped))

    # Convert to PIL Image
    rgba_img = Image.fromarray((colored * 255).astype(np.uint8), mode='RGBA')

    # Generate output filename automatically
    output_img_path = os.path.join(output_img_path, f"{model_type}_Prescription_chart.png")

    # Save image
    rgba_img.save(output_img_path)
    print(f"✅ Visualization image saved: {output_img_path}")

    # Create matplotlib visualization
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    # Display clipped data
    norm = mpl.colors.Normalize(vmin=np.nanmin(data_clipped), vmax=np.nanmax(data_clipped))
    im = ax.imshow(data_clipped, cmap=colormap, norm=norm)
    ax.set_title("Prediction Map", fontsize=12, pad=10)
    ax.axis("off")

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Prediction Value", fontsize=10)

    plt.tight_layout()
    return fig


def plot_percentile_clipping_hist(tif_path, clip_percentile=5):
    """
    Plots the histogram of a TIF image before and after clipping pixel values based on the given percentile.

    Parameters
    ----------
    tif_path : str
        Path to the TIF file.
    clip_percentile : float
        Percentile for clipping the data. Default is 5, meaning the lowest and highest 5% will be clipped.

    Returns
    ----------
    fig : matplotlib.figure.Figure
        The figure object that can be saved using `fig.savefig(...)`.

    Raises
    ----------
    ValueError
        If no valid data is available for histogram plotting.
    """

    # Read the first band from the TIF file
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)

    # Remove NaN or invalid values
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("❌ No valid data available for histogram plotting!")

    # Compute clipping bounds
    lower = np.percentile(valid, clip_percentile)
    upper = np.percentile(valid, 100 - clip_percentile)

    # Clip data
    data_clipped = np.clip(valid, lower, upper)

    # Automatically adjust x-axis range
    range_width = upper - lower
    x_range_padding = range_width * 0.3
    x_min = lower - x_range_padding
    x_max = upper + x_range_padding

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram of original data
    ax.hist(valid, bins=100, color='skyblue', alpha=0.6, label='Original Data', density=True)

    # Histogram of clipped data
    ax.hist(data_clipped, bins=100, color='orange', alpha=0.6, label='Clipped Data', density=True)

    # Add vertical lines for clipping thresholds
    ax.axvline(lower, color='red', linestyle='--', linewidth=1, label=f'Lower Bound: {lower:.2f}')
    ax.axvline(upper, color='green', linestyle='--', linewidth=1, label=f'Upper Bound: {upper:.2f}')

    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Density")
    ax.set_title(f"TIF Image Histogram Comparison (Clip Percentile = {clip_percentile}%)")
    ax.set_xlim(x_min, x_max)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    tif_path = r'D:\Code_Store\InversionSoftware\S3_prediction_vis\s3_Prediction_Results\0307.tif'
    output_img_path = r'D:\Code_Store\InversionSoftware\S3_prediction_vis\s3_Prediction_Results'
    model_type = 'Auto'
    colormap = 'RdYlGn'
    fig = visualize_tif(tif_path, output_img_path, model_type, colormap, show_colorbar=True, clip_percentile=5)
    fig.show()

    # Example usage:
    fig = plot_percentile_clipping_hist(tif_path, clip_percentile=5)
    fig.show()
