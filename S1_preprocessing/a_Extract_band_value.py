"""
===============================================================
Script Name: a_Extract_band_value.py

Script Function:
-----------------------------------------------
This script is the first step in data processing, used to extract
pixel values from single-band or multi-band remote sensing images (GeoTIFF)
based on given latitude and longitude coordinates or vector polygons.

Main functions include:
    - Extracting pixel values at single points from imagery (single or multi-band)
    - Extracting the mean pixel values in a window around a point
    - Extracting mean pixel values within vector polygons
    - Coordinate transformation: converting WGS84 (lat/lon) coordinates to
      the projected coordinate system of the imagery

Supported input file formats:
    - TIFF remote sensing images (.tif)
    - Point coordinate files: .csv, .xlsx, .txt, .shp (points)
    - Vector polygon files: .shp (Polygon or MultiPolygon)

Typical workflow:
-----------------------------------------------
1. Convert latitude and longitude coordinates to the projected
   coordinate system of the imagery
2. Extract the corresponding pixel values (or calculate area means)
3. Save the results to a table

Tips:
-----------------------------------------------
Using conda to install GDAL is very convenient:
conda install -c conda-forge gdal
===============================================================
"""

import rasterio
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import mask

def read_coordinates(file_path):
    """
    Read files containing latitude and longitude coordinates
    (supports CSV, Excel, TXT, SHP).

    :param file_path: path to the file
    :return: dataframe with columns 'lon' and 'lat'
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['lon', 'lat'])
    elif file_path.endswith('.shp'):
        gdf = gpd.read_file(file_path)
        df = pd.DataFrame({'lon': gdf.geometry.x, 'lat': gdf.geometry.y})
    else:
        raise ValueError("Unsupported file format. Use CSV, Excel, TXT, or SHP.")

    return df[['lon', 'lat']]

def convert_coordinates_to_tif_crs(tif_path, file_path):
    """
    Read the coordinate system of a TIFF image, and convert WGS84
    (latitude/longitude) coordinates in a file to the projection
    coordinate system of the image.

    Parameters:
        tif_path (str): path to the TIFF image file
        file_path (str): path to a file containing latitude and longitude in WGS84

    Returns:
        pandas.DataFrame: containing only transformed projected coordinates (lon, lat).
    """

    # 1. Read the CRS of the TIFF image
    with rasterio.open(tif_path) as dataset:
        target_crs = dataset.crs

    print(f"Image CRS: {target_crs}")

    # 2. Read the file into a dataframe
    df = read_coordinates(file_path)

    # Ensure the dataframe contains the required columns
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("The file is missing required columns 'lat' or 'lon'.")

    # 3. Coordinate transformation
    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    easting, northing = zip(*df.apply(lambda row: transformer.transform(row["lon"], row["lat"]), axis=1))

    # 4. Keep only transformed coordinates with updated column names
    df_transformed = pd.DataFrame({"lon": easting, "lat": northing})

    return df_transformed

def extract_values_from_tif1(tif_path, df_transformed):
    """
    Extract pixel values at specified projected coordinates (lon, lat)
    from a single-band TIFF image.

    Parameters:
        tif_path (str): path to the TIFF image
        df_transformed (pd.DataFrame): DataFrame containing 'lon' and 'lat'
                                       in geographic or projected coordinates.

    Returns:
        pd.DataFrame: containing original 'lon' and 'lat' and extracted 'value'.
    """

    with rasterio.open(tif_path) as dataset:
        values = [
            dataset.sample([(row["lon"], row["lat"])]).__next__()[0]
            for _, row in df_transformed.iterrows()
        ]

    df_transformed["value"] = values

    return df_transformed

def extract_values_from_tif2(tif_path, df_transformed):
    """
    Extract all band pixel values at specified projected coordinates
    (lon, lat) from a multi-band TIFF image, checking whether the
    coordinates fall within the image bounds.

    Parameters:
        tif_path (str): path to the TIFF image
        df_transformed (pd.DataFrame): DataFrame containing 'lon' and 'lat'
                                       in projected coordinates.

    Returns:
        pd.DataFrame: containing original 'lon' and 'lat' and the pixel
                      values from each band (band1-value, band2-value, ...).
                      Points outside the image bounds are marked as None.
    """

    with rasterio.open(tif_path) as dataset:
        num_bands = dataset.count
        bounds = dataset.bounds

        extracted_values = []

        for idx, row in df_transformed.iterrows():
            lon, lat = row["lon"], row["lat"]

            if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
                print(f"⚠️ Point {idx+1} (lon={lon}, lat={lat}) is outside the image bounds!")
                extracted_values.append([None] * num_bands)
                continue

            pixel_values = next(dataset.sample([(lon, lat)]), [None] * num_bands)
            extracted_values.append(pixel_values)

    bands_df = pd.DataFrame(extracted_values, columns=[f"band{i+1}-value" for i in range(num_bands)])
    df_result = df_transformed.copy()
    df_result = pd.concat([df_result, bands_df], axis=1)

    return df_result

def extract_values_from_tif3(tif_path, df_transformed, window_size=10):
    """
    Extract the mean values of all bands from a window of size
    window_size around each specified projected coordinate (lon, lat)
    in a multi-band TIFF image.

    Parameters:
        tif_path (str): path to the TIFF image
        df_transformed (pd.DataFrame): DataFrame containing 'lon' and 'lat'
                                       in projected coordinates.
        window_size (int): size of the square window (e.g. 10 means 10×10).
                           Default is 10.

    Returns:
        pd.DataFrame: containing 'lon', 'lat' and the mean value of each band
                      (band1, band2, ...). Points outside the image bounds
                      are marked as NaN.
    """
    with rasterio.open(tif_path) as dataset:
        num_bands = dataset.count
        bounds = dataset.bounds
        transform = dataset.transform

        all_bands = dataset.read()  # shape: (num_bands, height, width)

        height, width = dataset.height, dataset.width
        half_window_size = window_size // 2

        def process_point(row):
            lon, lat = row["lon"], row["lat"]

            if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
                print(f"⚠️ Coordinate (lon={lon}, lat={lat}) is outside the image bounds!")
                return [np.nan] * num_bands

            col, row_idx = ~transform * (lon, lat)
            col, row_idx = int(round(col)), int(round(row_idx))

            col_start = max(col - half_window_size, 0)
            row_start = max(row_idx - half_window_size, 0)
            col_end = min(col + half_window_size + 1, width)
            row_end = min(row_idx + half_window_size + 1, height)

            window_means = np.nanmean(all_bands[:, row_start:row_end, col_start:col_end], axis=(1, 2))

            return window_means

        extracted_means = np.array(df_transformed.apply(process_point, axis=1).tolist())

    bands_df = pd.DataFrame(extracted_means, columns=[f"band{i + 1}" for i in range(num_bands)])
    df_result = pd.concat([df_transformed, bands_df], axis=1)

    return df_result

def extract_multiband_mean_by_shapefile(tif_path, shp_path):
    """
    Extract the mean pixel values of each band from a multi-band
    GeoTIFF image within polygons defined in a shapefile.

    The output columns are named as band1-value, band2-value, etc.

    Parameters:
    ----------
    tif_path : str
        Path to the multi-band GeoTIFF image.
    shp_path : str
        Path to the shapefile containing polygon boundaries.

    Returns:
    ----------
    pd.DataFrame
        One row per polygon, with columns for the mean of each band.
    """
    gdf = gpd.read_file(shp_path)

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        band_count = src.count
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        results = []

        for idx, geom in enumerate(gdf.geometry):
            try:
                out_image, _ = mask.mask(src, [geom], crop=True)
                band_means = {}

                for i in range(band_count):
                    band_data = out_image[i]
                    valid = band_data[band_data != src.nodata]
                    mean_val = np.mean(valid) if valid.size > 0 else np.nan
                    band_means[f'band{i+1}-value'] = mean_val

                results.append(band_means)

            except Exception as e:
                print(f"❌ Polygon {idx} extraction failed: {e}")
                results.append({f'band{i+1}-value': np.nan for i in range(band_count)})

    result_df = pd.DataFrame(results)
    result_df.insert(0, 'index', range(len(result_df)))

    return result_df

def Extract_band_value_main(tif_path, file_path, windowsize=10):
    """
    Main function for extracting spectral information: extracts pixel
    values or mean values from a multi-band remote sensing image
    corresponding to input coordinates or shapefile polygons.

    Parameters:
    ----------
    tif_path : str
        Path to the input multi-band GeoTIFF image.
    file_path : str
        Path to the sampling points or polygon boundaries.
        Supported formats: .shp (points/polygons), .xlsx, .xls, .csv, .txt
        (with latitude/longitude coordinates).
    windowsize : int, optional, default=10
        Size of the moving window (in pixels) used to calculate
        average values around points. Not used for polygon shapefiles.

    Returns:
    ----------
    df : pandas.DataFrame
        Table containing extracted values for each polygon or sample point:
        - For polygons: each row corresponds to one polygon, with columns
          band{i}-value
        - For points: each row corresponds to one point, with band values
          and transformed coordinates
    """
    if file_path.lower().endswith('.shp'):
        gdf = gpd.read_file(file_path)
        geom_types = gdf.geometry.geom_type.unique()

        if all(t in ['Polygon', 'MultiPolygon'] for t in geom_types):
            df = extract_multiband_mean_by_shapefile(tif_path, file_path)
            return df
        elif all(t in ['Point', 'MultiPoint'] for t in geom_types):
            df_transformed = convert_coordinates_to_tif_crs(tif_path, file_path)
            df = extract_values_from_tif3(tif_path, df_transformed, windowsize)
            return df
    else:
        df_transformed = convert_coordinates_to_tif_crs(tif_path, file_path)
        df = extract_values_from_tif3(tif_path, df_transformed, windowsize)
        return df

# ======= Example Run =======
if __name__ == "__main__":

    # Example: extract mean values from a window around sample points in a multi-band image
    tif_path = "testdata/20240525comp.tif"       # Path to the TIFF image
    file_path = "testdata/20240525xy+N.xlsx"     # File containing WGS84 latitude/longitude
    output_path = "D:/your_folder/extracted_values_output.xlsx"

    # 1. Coordinate transformation
    df_transformed = convert_coordinates_to_tif_crs(tif_path, file_path)
    print("✅ Coordinate transformation completed:")
    print(df_transformed.head())

    # 2. Extract mean values from a region in the multi-band image
    df_result = extract_values_from_tif3(tif_path, df_transformed, window_size=10)
    df_result.to_excel(output_path, index=False)

    print("✅ Pixel value extraction completed and saved to:", output_path)
