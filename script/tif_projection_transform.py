"""
Module Name: tif_projection_transform

This script performs projection transformation (reprojection) for GeoTIFF remote sensing images,
converting the image from its original coordinate system to a specified target CRS.

Main Features:
- Automatically calculates the new image size and affine transform needed for reprojection.
- Resamples the original image into the target coordinate system.
- Supports multi-band data.
- Allows different resampling methods (default: nearest neighbor).

Inputs:
- Path to the original GeoTIFF file
- Target projection CRS (EPSG code)
- Path to the output file

Outputs:
- Reprojected GeoTIFF file

Dependencies:
- rasterio
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_tif(src_path, dst_path, dst_crs='EPSG:4326', resampling_method=Resampling.nearest):
    """
    Reprojects a GeoTIFF file to a new coordinate system.

    Parameters:
        src_path (str): Path to the input GeoTIFF file.
        dst_path (str): Path to save the reprojected GeoTIFF file.
        dst_crs (str): Target coordinate reference system (default is WGS84, EPSG:4326).
        resampling_method (rasterio.warp.Resampling): Resampling method.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )
    print(f"✅ Reprojection completed. Output file: {dst_path}")

if __name__ == "__main__":
    # Example usage
    src_path = r"D:\Zyh\20250307_index_ndvi.tif"
    dst_path = "output_geographic.tif"
    reproject_tif(src_path, dst_path, dst_crs='EPSG:4326')
