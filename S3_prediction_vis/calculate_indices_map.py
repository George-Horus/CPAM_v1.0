"""
===============================================================
Module Name: calculate_indices_map

This module extracts and computes common vegetation indices
from multi-band remote sensing imagery (GeoTIFF) and outputs:
    - GeoTIFF files for vegetation indices
    - Vegetation index histograms (PIL format)

Main Features:
1. Reads spectral bands from a multi-band remote sensing image,
   with simple normalization applied.
2. Calls VMID_1 module tools to batch-compute multiple vegetation indices
   like NDVI, NDRE, etc.
3. Saves each single-band vegetation index result as a GeoTIFF file,
   preserving spatial reference information.
4. Generates and outputs grayscale histograms of vegetation indices
   for quick data distribution inspection.

Core functions:
- predict_from_tif: computes specified vegetation indices and returns
                    results with image metadata.
- save_array_as_tif: saves a 2D array as a GeoTIFF file.
- generate_histogram_from_tif: generates a histogram image (PIL)
                               from a single-band GeoTIFF.
- indices_map_main: executes the full workflow to produce index images
                    and histograms.

Inputs:
- Multi-band GeoTIFF file path
- List of band names (in the order matching the image bands)
- List of vegetation index names to compute

Outputs:
- GeoTIFF file path for vegetation indices
- Histogram of the vegetation index (PIL Image)

"""

import os
from S1_preprocessing import VMID_1
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def predict_from_tif(multi_tif_data_path, bandlist, index_list):
    """
    Computes specified vegetation indices from a multi-band
    spectral remote sensing image.

    Parameters
    ----------
    multi_tif_data_path : str
        Path to input remote sensing image (GeoTIFF format
        containing multiple bands).
    bandlist : list[str]
        Ordered list of names for each image band,
        e.g. ['blue', 'green', 'nir', 'red', 'rededge'].
    index_list : list[str]
        Vegetation index names to compute,
        e.g. ['NDVI', 'NDRE'].

    Returns
    ----------
    indices : dict[str, np.ndarray]
        Dictionary mapping vegetation index names to computed
        2D arrays (image shape).
    meta : dict
        Metadata from the original image, used for writing
        GeoTIFFs later.
    """

    # 1. Load the image and read each band
    standard_bands = ['blue', 'green', 'red', 'nir', 'rededge']
    band_data = {band: None for band in standard_bands}

    # Band matching for vegetation index calculations
    with rasterio.open(multi_tif_data_path) as src:
        if len(bandlist) != src.count:
            raise ValueError(
                f"[Error] bandlist length ({len(bandlist)}) "
                f"does not match the number of image bands ({src.count})."
            )
        for i, band_name in enumerate(bandlist):
            name = band_name.lower()
            if name in band_data:
                data = src.read(i + 1).astype(np.float32)
                data[(data < 0) | (data > 1)] = 0  # Simple normalization
                band_data[name] = data
        meta = src.meta

    # 2. Get the image dimensions
    for band in band_data.values():
        if band is not None:
            rows, cols = band.shape
            break
    else:
        raise ValueError("❌ Unable to determine image size; all bands are None.")

    # 3. Compute vegetation indices using the VMID index library
    vminstance = VMID_1.VMID(
        red=band_data['red'],
        green=band_data['green'],
        blue=band_data['blue'],
        red_edge=band_data['rededge'],
        nir=band_data['nir'],
        index_list=index_list
    )
    vminstance._calculate_indices(index_list)
    index_results = vminstance.indices

    print("Number of successfully computed indices:", len(index_results))
    print("List of successfully computed indices:", list(index_results.keys()))

    if not index_results:
        raise ValueError("❌ No vegetation indices could be computed. Check the band data or index_list.")

    return index_results, meta

def save_array_as_tif(array, reference_meta, save_dir, indice_name):
    """
    Saves a computed 2D array as a GeoTIFF image.

    Parameters
    ----------
    array : ndarray
        Computed vegetation index 2D array.
    reference_meta : dict
        Metadata from the reference image to preserve spatial reference.
    save_dir : str
        Output folder path.
    indice_name : str
        Name of the vegetation index for the output filename.

    Returns
    ----------
    full_path : str
        Full path to the saved GeoTIFF file.
    """
    # Build the save path
    os.makedirs(save_dir, exist_ok=True)
    tif_path = os.path.join(save_dir, f'{indice_name}.tif')

    # Update metadata
    meta = reference_meta.copy()
    meta.update({
        'count': 1,
        'dtype': 'float32'
    })

    # Save the image
    with rasterio.open(tif_path, 'w', **meta) as dst:
        dst.write(array.astype(np.float32), 1)

    print(f"✅ {indice_name} index map saved to: {tif_path}")
    return tif_path

def generate_histogram_from_tif(tif_path):
    """
    Generates a grayscale histogram image (PIL format)
    from a single-band GeoTIFF.

    Parameters
    ----------
    tif_path : str
        Path to the input single-band GeoTIFF file.

    Returns
    ----------
    hist_img : PIL.Image.Image
        The generated histogram as a PIL image object.
    """
    # 1. Read the image data
    with rasterio.open(tif_path) as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata

    # 2. Clean invalid pixel values
    if nodata is not None:
        array[array == nodata] = np.nan
    array[array < 0] = np.nan  # Adjust if negative values should be retained

    # 3. Extract valid pixel values
    valid_values = array[~np.isnan(array)]

    # 4. Check if data is available
    if valid_values.size == 0:
        raise ValueError("❌ No valid pixels in the input image. Cannot generate histogram.")

    # 5. Create the histogram
    plt.figure(figsize=(8, 4))
    plt.hist(valid_values, bins=256, color='blue', alpha=0.7)
    plt.title("Gray Histogram", fontsize=14)
    plt.xlabel("Pixel Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().tick_params(width=1.5)

    # 6. Save to an in-memory image (PIL)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    hist_img = Image.open(buf)

    return hist_img

def indices_map_main(multi_tif_data_path, bandlist, index_list, output_dir):
    import os

    # Set output directory
    save_path = os.path.join(output_dir, 'indices_map')
    os.makedirs(save_path, exist_ok=True)

    # Compute vegetation indices
    indices, meta = predict_from_tif(multi_tif_data_path, bandlist, index_list)

    # Retrieve the first index and its array
    indice_name, array = next(iter(indices.items()))

    # Save GeoTIFF file
    tif_path = save_array_as_tif(array, meta, save_path, indice_name)

    # Generate histogram
    hist_img = generate_histogram_from_tif(tif_path)

    return tif_path, hist_img

# Example usage
if __name__ == '__main__':
    # Input parameters
    multi_tif_path = '1.tif'
    bands = ['blue', 'green', 'nir', 'red', 'rededge']
    indices_to_compute = ['NDVI', 'NDRE']
    output_directory = 'D:\Code_Store\Water_content_inversion'

    # Generate specified vegetation index GeoTIFF
    generated_tif_paths = indices_map_main(
        multi_tif_path,
        bands,
        indices_to_compute,
        output_directory
    )
    print("All generated GeoTIFF file paths:")
    for path in generated_tif_paths:
        print(path)
