"""
Module Name: shp_excel_tools

This module provides tools for converting between Shapefiles and Excel files,
suitable for geospatial data processing and analysis scenarios.

Main Features:
1. Extract Point features and attributes from a Shapefile and export them to an Excel spreadsheet.
2. Create a Shapefile from an Excel file containing coordinate columns and other attributes.

Main Functions:
- shp_points_to_excel:
    Exports Point features and attributes from a Shapefile to an Excel file.
- excel_to_shapefile:
    Converts coordinates and attributes in an Excel file to a Shapefile.

Inputs:
- Shapefile path
- Excel file path
- Coordinate column names
- EPSG code for coordinate system

Outputs:
- Excel file (.xlsx)
- Shapefile (.shp)
"""

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

def shp_points_to_excel(shp_path, excel_path):
    """
    Extract all point features from a Shapefile and save them to an Excel file.

    Parameters:
        shp_path (str): Path to the input .shp file
        excel_path (str): Path to the output .xlsx file
    """
    # Read the shapefile (supports multiple geometry types)
    gdf = gpd.read_file(shp_path)

    # Filter only Point geometries (others like polygons or lines will be ignored)
    gdf_points = gdf[gdf.geometry.type == "Point"].copy()

    if gdf_points.empty:
        print("⚠️ No Point features found in the input Shapefile.")
        return

    # Extract longitude and latitude (or projected coordinates)
    gdf_points["x"] = gdf_points.geometry.x
    gdf_points["y"] = gdf_points.geometry.y

    # Reset index (to preserve original order)
    gdf_points.reset_index(drop=True, inplace=True)

    # Save to Excel (including all attribute columns and coordinate columns)
    gdf_points.to_excel(excel_path, index=False)
    print(f"✅ Successfully exported {len(gdf_points)} points to: {excel_path}")

def excel_to_shapefile(excel_path, shp_output_path, lon_col="lon", lat_col="lat", crs_epsg=32650):
    """
    Convert an Excel file to a Shapefile.

    Parameters:
        excel_path (str): Path to the Excel file
        shp_output_path (str): Path to the output .shp file
        lon_col (str): Column name for longitude or X coordinate
        lat_col (str): Column name for latitude or Y coordinate
        crs_epsg (int): EPSG code for the coordinate reference system.
                        Default is 32650 (WGS 84 / UTM zone 50N).
                        Use 4326 for latitude/longitude.
    """
    df = pd.read_excel(excel_path)

    if not {lon_col, lat_col}.issubset(df.columns):
        raise ValueError(f"The Excel file must contain columns: {lon_col} and {lat_col}")

    # Create point geometries
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{crs_epsg}")

    # Save as Shapefile
    gdf.to_file(shp_output_path, driver='ESRI Shapefile')
    print(f"✅ Shapefile successfully saved to: {shp_output_path}")


excel_to_shapefile(
    excel_path=r"D:\Zyh\0317.xlsx",
    shp_output_path=r"D:\Zyh\0317_points.shp",
    lon_col="lon",
    lat_col="lat",
    crs_epsg=4326  # Use 4326 for latitude/longitude coordinates
)

if __name__ == "__main__":
    # Example: Convert Excel to Shapefile
    excel_to_shapefile(
        excel_path=r"0317.xlsx",
        shp_output_path=r"0317_points.shp",
        lon_col="lon",
        lat_col="lat",
        crs_epsg=4326
    )

    # Example: Extract Point features from Shapefile and export to Excel
    shp_path = r"polygon_centroids.shp"
    excel_path = r"output_points.xlsx"
    shp_points_to_excel(shp_path, excel_path)
