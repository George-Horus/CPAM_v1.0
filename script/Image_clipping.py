"""
Module Name: Image_clipping

This module is used for cropping GeoTIFF remote sensing images.
It supports two cropping modes:
    1. Crop a square of specified size from the center of the image.
    2. Crop the image using vector boundaries from a Shapefile.

Main Functions:
- crop_center_tif:
    Crops a specified area from the center of a GeoTIFF image while preserving georeferencing information.
- crop_tif_by_shapefile:
    Crops a GeoTIFF image based on vector boundaries defined in a Shapefile.

Inputs:
- Path to the GeoTIFF file
- Cropping size (in pixels)
- Path to the Shapefile (for vector-based cropping)
- Path to save the output file

Outputs:
- Cropped GeoTIFF file
"""

import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd

def crop_center_tif(input_tif_path, output_tif_path, crop_size=2000):
    """
    Crop a GeoTIFF image from its center with the specified size,
    while preserving georeferencing.

    Parameters:
        input_tif_path (str): Path to the input GeoTIFF file.
        output_tif_path (str): Path to save the cropped GeoTIFF file.
        crop_size (int): Size of the square region to crop (in pixels).
    """
    with rasterio.open(input_tif_path) as src:
        width, height = src.width, src.height
        bands = src.count

        center_x = width // 2
        center_y = height // 2

        left = max(center_x - crop_size // 2, 0)
        top = max(center_y - crop_size // 2, 0)

        right = min(left + crop_size, width)
        bottom = min(top + crop_size, height)

        win_width = right - left
        win_height = bottom - top

        window = Window(left, top, win_width, win_height)
        transform = src.window_transform(window)

        cropped_data = src.read(window=window)

        profile = src.profile
        profile.update({
            'height': win_height,
            'width': win_width,
            'transform': transform
        })

        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(cropped_data)

        print(f"✅ Successfully cropped center region and saved to: {output_tif_path}")


def crop_tif_by_shapefile(input_tif_path, shapefile_path, output_tif_path):
    """
    Crop a GeoTIFF image based on vector boundaries from a shapefile.
    Retains coordinate reference and clips the region to the mask.

    Parameters:
        input_tif_path (str): Path to the input GeoTIFF file.
        shapefile_path (str): Path to the Shapefile defining vector boundaries.
        output_tif_path (str): Path to save the cropped GeoTIFF file.
    """
    # Read the vector boundaries
    shapes = gpd.read_file(shapefile_path)
    with rasterio.open(input_tif_path) as src:
        crs = src.crs.to_epsg()
    shapes = shapes.to_crs(epsg=crs)  # Ensure CRS consistency

    geoms = shapes.geometry.values
    geoms = [geom.__geo_interface__ for geom in geoms]  # Convert to GeoJSON format

    with rasterio.open(input_tif_path) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_tif_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"✅ Successfully cropped by shapefile and saved to: {output_tif_path}")


if __name__ == "__main__":
    input_tif = r'D:\20250307ROI.tif'

    # Method 1: Center crop
    output_center_tif = '0307cropped_center_1024x1024.tif'
    crop_center_tif(input_tif, output_center_tif, crop_size=1024)

    # # Method 2: Crop by shapefile
    # shapefile_path = r'D:\Code_Store\InversionSoftware\script\UAV_image\ROI.shp'
    # output_shp_crop_tif = r'D:\Code_Store\InversionSoftware\script\UAV_image\20250307.tif'
    # crop_tif_by_shapefile(input_tif, shapefile_path, output_shp_crop_tif)
