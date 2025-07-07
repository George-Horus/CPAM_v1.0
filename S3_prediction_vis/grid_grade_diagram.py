"""
Module name: grid_grade_diagram

This module is designed for agricultural remote sensing analysis.
It calculates nitrogen fertilizer prescription maps (AGN) based on nitrogen concentration prediction results
and generates corresponding grid-based classification visualizations.

Main features:
1. Compute AGN (additional fertilizer amount in kg/ha) from a nitrogen concentration GeoTIFF image.
2. Save AGN results as GeoTIFF files.
3. Perform block statistics on the AGN image, divide into grids (e.g. 2m × 2m), and generate a grid-based prescription map.
4. Output a colorful PNG image, showing clear fertilizer levels across different regions.
5. Provide a fully encapsulated main workflow function (grid_main) for generating AGN images and grid maps with one click.

Main functions:
- calculate_agn_from_tif: Computes the fertilizer prescription map from a nitrogen concentration map and saves it as GeoTIFF.
- visualize_tif_with_grid_levels: Displays an AGN GeoTIFF as a classified color grid map.
- grid_main: Wraps the entire process to generate both GeoTIFF and PNG grid maps from a nitrogen concentration map.

Inputs:
- Nitrogen concentration prediction image (GeoTIFF format)
- Parameters like standard nitrogen concentration, standard crop biomass, fertilizer solution concentration, fertilizer utilization rate
- Grid width (in meters)
- Output directory

Outputs:
- GeoTIFF file of fertilizer prescription map
- PNG grid-based prescription map
- matplotlib Figure object (for GUI or other display)
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import os

def calculate_agn_from_tif(
    nitrogen_tif_path,
    output_tif_path=None,
    N_std=16.5,    # Standard nitrogen concentration (g/kg)
    B_std=3000,    # Standard crop biomass (kg/ha)
    n=1,           # Fertilizer solution concentration (mg/mL)
    C_std=1,       # Vegetation coverage in the standard region
    C_x=1,         # Vegetation coverage in the current region (may be fixed)
    k=0.4,         # Fertilizer utilization rate
    u=1            # Fertilizer conversion efficiency
):
    """
    Compute AGN (additional nitrogen fertilizer amount) based on a nitrogen concentration image
    and save it as a GeoTIFF file.

    Parameters:
        nitrogen_tif_path (str): Path to the nitrogen concentration prediction image.
        output_tif_path (str or None): Path to save the GeoTIFF. If None, auto-generate filename.
        N_std (float): Standard nitrogen concentration.
        B_std (float): Standard crop biomass.
        n (float): Fertilizer solution concentration.
        C_std (float): Vegetation coverage in standard region.
        C_x (float): Vegetation coverage in the current region.
        k (float): Fertilizer utilization rate.
        u (float): Fertilizer conversion efficiency.

    Returns:
        output_tif_path (str): Path to the saved AGN GeoTIFF file.
    """
    # Auto-generate output filename if none provided
    if output_tif_path is None:
        base_name = os.path.splitext(os.path.basename(nitrogen_tif_path))[0]
        output_tif_path = f"{base_name}_AGN.tif"

    # Open nitrogen concentration prediction image
    with rasterio.open(nitrogen_tif_path) as src:
        nitrogen = src.read(1).astype(np.float32)
        meta = src.meta.copy()

    # Handle invalid values
    nitrogen[nitrogen <= 0] = np.nan

    # Compute nitrogen deficiency (Nx)
    Nx = N_std - nitrogen
    Nx[Nx < 0] = 0  # No additional fertilizer needed if nitrogen is sufficient

    # Compute AGN (kg/ha)
    numerator = Nx * n * B_std * C_std
    denominator = k * u * C_x
    AGN = (numerator / denominator) / 1000  # Convert from g/ha to kg/ha

    # Replace NaN with -1 to indicate nodata
    AGN_out = np.where(np.isnan(AGN), -1, AGN).astype(np.float32)

    # Update metadata
    meta.update({
        "dtype": "float32",
        "nodata": -1,
        "count": 1
    })

    # Save GeoTIFF output
    with rasterio.open(output_tif_path, 'w', **meta) as dst:
        dst.write(AGN_out, 1)

    print(f"✅ AGN prescription map saved to: {output_tif_path}")

    return output_tif_path


def visualize_tif_with_grid_levels(tif_file, grid_width_meters=2, save_path=None, show_plot=False):
    """
    Visualize a TIFF image as a classified grid map, save as PNG, and return the matplotlib Figure object.

    Parameters:
        tif_file (str): Input TIFF file path.
        grid_width_meters (float): Grid cell width in meters.
        save_path (str or None): PNG save path. If None, generate filename automatically.
        show_plot (bool): Whether to display the plot (plt.show()).

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        save_path (str): Path to the saved PNG file.
    """
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    with rasterio.open(tif_file) as src:
        data = src.read(1).astype(np.float32)
        pixel_width, pixel_height = src.res
        nodata = src.nodata

    # Replace invalid values
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data < 0] = np.nan

    valid_pixels = data[~np.isnan(data)]
    if valid_pixels.size == 0:
        raise ValueError("❌ No valid data found. Cannot proceed.")

    # Calculate percentile-based classification boundaries
    level_boundaries = np.percentile(valid_pixels, np.linspace(0, 100, 6))
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    cmap = ListedColormap(colors)

    # Grid dimensions
    grid_width_pixels = int(grid_width_meters / pixel_width)
    height, width = data.shape
    grid_rows = height // grid_width_pixels
    grid_cols = width // grid_width_pixels

    # Initialize empty grid map
    mapped_grid = np.full_like(data, np.nan, dtype=np.float32)
    for i in range(grid_rows):
        for j in range(grid_cols):
            r0 = i * grid_width_pixels
            r1 = min((i + 1) * grid_width_pixels, height)
            c0 = j * grid_width_pixels
            c1 = min((j + 1) * grid_width_pixels, width)

            block = data[r0:r1, c0:c1]
            mean_val = np.nanmean(block)
            cls = np.digitize(mean_val, level_boundaries) - 1 if not np.isnan(mean_val) else np.nan
            mapped_grid[r0:r1, c0:c1] = cls

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    im = ax.imshow(mapped_grid, cmap=cmap, vmin=0, vmax=4)

    level_labels = [f"{level_boundaries[i]:.2f}–{level_boundaries[i + 1]:.2f}" for i in range(5)]
    cbar = fig.colorbar(im, ax=ax, ticks=range(5))
    cbar.ax.set_yticklabels(level_labels, fontsize=10, fontweight='bold')
    cbar.set_label("Grade interval", fontsize=12, fontweight='bold')

    ax.set_title(f"Grid prescription map (Grid width {grid_width_meters} m)", fontsize=14, fontweight='bold')

    # Add grid lines
    for i in range(1, grid_rows):
        ax.axhline(i * grid_width_pixels, color='black', linewidth=0.5)
    for j in range(1, grid_cols):
        ax.axvline(j * grid_width_pixels, color='black', linewidth=0.5)

    # Generate PNG save path automatically if needed
    if save_path is None:
        tif_dir = os.path.dirname(tif_file)
        base_name = os.path.splitext(os.path.basename(tif_file))[0]
        save_path = os.path.join(tif_dir, f"{base_name}_grid_{int(grid_width_meters)}m.png")

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ PNG grid map saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, save_path


def grid_main(nitrogen_tif_path, grid_width_meters, N_std, B_std, output_dir):
    """
    Main workflow:
        - Computes AGN GeoTIFF from nitrogen concentration map.
        - Generates and saves PNG grid visualization.

    Parameters:
        nitrogen_tif_path (str): Path to nitrogen concentration prediction image.
        grid_width_meters (float): Grid width in meters.
        N_std (float): Standard nitrogen concentration.
        B_std (float): Standard crop biomass.
        output_dir (str): Root directory for output files.

    Returns:
        fig (matplotlib.figure.Figure): The grid map figure.
        png_save_path (str): Path to the saved PNG file.
    """
    print(nitrogen_tif_path, grid_width_meters, N_std, B_std, output_dir)
    save_dir = os.path.join(output_dir, 'grid_map')
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(nitrogen_tif_path))[0]
    output_tif_path = os.path.join(save_dir, f"{base_name}_AGN.tif")

    output_tif_path = calculate_agn_from_tif(
        nitrogen_tif_path,
        output_tif_path,
        N_std=N_std,
        B_std=B_std
    )

    png_save_path = os.path.join(save_dir, f"{base_name}_grid_{int(grid_width_meters)}m.png")

    fig, _ = visualize_tif_with_grid_levels(
        output_tif_path,
        grid_width_meters=grid_width_meters,
        save_path=png_save_path
    )

    return fig, png_save_path


# ==== Example usage ====
if __name__ == "__main__":
    tif_path = r"0331.tif"
    visualize_tif_with_grid_levels(tif_path, grid_width_meters=2)
