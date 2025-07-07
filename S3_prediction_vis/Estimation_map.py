"""
===============================================================
Module Name: Estimation_map

This module performs large-scale prediction and mapping on
remote sensing imagery. Core functionality includes:

    - Calculating various vegetation indices from multi-band
      remote sensing images (GeoTIFF).
    - Pixel-wise prediction on imagery using different types
      of models (ML / DL / AutoML).
    - Supports tile-based processing for large images,
      conserving memory.
    - Outputs prediction results as GeoTIFF images.
    - Automatically generates and saves histograms of
      prediction results.
    - Provides real-time memory monitoring and logging for
      efficient performance tuning during batch processing.

Supported model types:
    - Traditional machine learning models (.joblib),
      e.g. SVR, RF, GBDT, XGBoost, KNN, etc.
    - Deep learning models (.keras),
      e.g. CNN, LSTM, GRU, DNN, AutoEncoder, etc.
    - AutoML models, e.g. AutoGluon or TPOT.

Main workflow:
    1. Load the user-specified model.
    2. Read the multi-band remote sensing imagery.
    3. Compute user-specified vegetation indices.
    4. Perform tile-based, pixel-wise prediction on the imagery.
    5. Save prediction results as GeoTIFF files.
    6. Generate and save histograms of the prediction results
       as PNG files.
    7. Output paths to all generated files and histogram images.

Main functions:
- load_model: Loads a model based on the specified type.
- safe_predict: Generic prediction interface supporting ML,
                DL, and AutoML.
- predict_from_tif_tile_based: Tile-based prediction for
                                large imagery.
- save_array_as_tif: Saves prediction results as GeoTIFF.
- Estimation_map_main: Master function controlling the
                       complete prediction workflow.

Inputs:
- Path to remote sensing imagery (GeoTIFF).
- Path to model file (.joblib, .keras, AutoGluon directory).
- Choice of model type (ML / DL / Auto).
- Band list.
- Vegetation index list.
- Output directory.

Outputs:
- GeoTIFF file of prediction map.
- Histogram of predictions (PNG format).
- PIL image object of the histogram.
- Log file.
"""
from S1_preprocessing import VMID_1
import joblib
import rasterio
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from PIL import Image
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from rasterio.windows import Window
from autogluon.tabular import TabularPredictor
from tpot import TPOTRegressor
from S3_prediction_vis import visualize_tif
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_model(model_path, modeltype):
    """
    Loads various types of machine learning or deep learning models.

    Supported formats include:
    - Traditional models (.joblib): SVR, RF, KNN, MLR, GBDT,
      XGBoost, BPNN, etc.
    - Deep learning models (.keras): CNN, LSTM, GRU, DNN,
      AutoEncoder, etc.
    - AutoGluon models (directory names containing autogluon)
    - TPOT models (directory or filenames containing tpot)

    Returns:
    ----------
    model : object
        The loaded model object ready for prediction.
    model_type : str
        Type of the model for subsequent processing logic.
    """
    print(model_path)
    if model_path.endswith('.joblib') and modeltype == 'ML':
        model = joblib.load(model_path)
        model_type = 'ML'
    elif model_path.endswith('.keras') and modeltype == 'DL':
        model = tf.keras.models.load_model(model_path)
        model_type = 'DL'
    elif model_path.endswith('.pkl') and modeltype == 'Auto':
        model = joblib.load(model_path)
        model_type = 'tpot'
    elif model_path.endswith('AutoGluon') and modeltype == 'Auto':
        model = TabularPredictor.load(model_path)
        model_type = 'autogluon'
    else:
        raise ValueError(f"❌ Unsupported model format: {model_path}")
    print(f"✅ Model loaded successfully: {model_type} type")
    return model, model_type

def safe_predict(model, X, model_type=None, feature_names=None, verbose=0):
    """
    Generic prediction interface compatible with multiple model types
    (ML, DL, AutoML).

    Parameters:
    ----------
    model : object
        Loaded model object.
    X : ndarray or pd.DataFrame
        Input feature array.
    model_type : str
        String identifier for model type to determine handling logic.
    feature_names : list[str]
        Required if input is an array and the model is AutoGluon.
    verbose : int
        Verbosity level for Keras prediction.

    Returns:
    ----------
    y_pred : ndarray
        Flattened 1D prediction array.
    """

    if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return model.predict(np.array(X), verbose=verbose).flatten()

    elif isinstance(model, TabularPredictor):
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                raise ValueError("AutoGluon requires feature_names to build a DataFrame")
            X = pd.DataFrame(X, columns=feature_names)
        return model.predict(X).to_numpy().flatten()

    elif isinstance(model, TPOTRegressor):
        return model.predict(X).flatten()

    else:
        if hasattr(X, 'values'):
            X_input = X.values
        else:
            X_input = np.array(X)
        return model.predict(X_input).flatten()

def predict_from_tif_tile_based(tif_path, model, model_type, bandlist, index_list, tile_size=512, batch_size=100_000):
    """
    Tile-based prediction function with progress bar for large
    remote sensing imagery.

    This enables processing of high-resolution or large-area images
    while keeping memory usage manageable.

    Parameters:
    ----------
    tif_path : str
        Path to the GeoTIFF image.
    model : object
        Loaded model object.
    model_type : str
        Type of the model.
    bandlist : list[str]
        List of band names matching the order of bands in the TIFF.
    index_list : list[str]
        List of vegetation indices to compute.
    tile_size : int
        Size of the tiles in pixels.
    batch_size : int
        Number of pixels processed in one prediction batch.

    Returns:
    ----------
    y_pred_final : ndarray
        Predicted values for the entire image as a 2D array.
    meta : dict
        Image metadata for saving GeoTIFF.
    """
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        meta = src.meta.copy()

        standard_bands = ['blue', 'green', 'red', 'nir', 'rededge']
        band_indices = {bandlist[i].lower(): i + 1 for i in range(len(bandlist)) if bandlist[i].lower() in standard_bands}

        y_pred_final = np.zeros((height, width), dtype=np.float32)

        y_tiles = range(0, height, tile_size)
        x_tiles = range(0, width, tile_size)
        total_tiles = len(y_tiles) * len(x_tiles)

        with tqdm(total=total_tiles, desc="📦 Performing tile-based prediction on imagery") as pbar:
            for y in y_tiles:
                for x in x_tiles:
                    win = Window(x, y, min(tile_size, width - x), min(tile_size, height - y))

                    band_data = {}
                    for band_name, idx in band_indices.items():
                        band = src.read(idx, window=win).astype(np.float32)
                        band[(band < 0) | (band > 1)] = 0
                        band_data[band_name] = band

                    if not band_data:
                        pbar.update(1)
                        continue

                    h, w = next(iter(band_data.values())).shape

                    vminstance = VMID_1.VMID(
                        red=band_data.get('red'),
                        green=band_data.get('green'),
                        blue=band_data.get('blue'),
                        red_edge=band_data.get('rededge'),
                        nir=band_data.get('nir'),
                        index_list=index_list
                    )
                    vminstance._calculate_indices(index_list)
                    index_results = vminstance.indices

                    if not index_results:
                        pbar.update(1)
                        continue

                    feature_names = list(index_results.keys())
                    feature_stack = np.stack(list(index_results.values()), axis=-1)
                    X = feature_stack.reshape(-1, feature_stack.shape[-1])

                    valid_mask = np.isfinite(X).all(axis=1)
                    X_valid = X[valid_mask]

                    y_pred_flat = np.zeros(X.shape[0], dtype=np.float32)

                    if len(X_valid) > 0:
                        preds = []
                        for i in range(0, len(X_valid), batch_size):
                            batch = pd.DataFrame(X_valid[i:i+batch_size], columns=feature_names)
                            batch_pred = safe_predict(model, batch, model_type=model_type, feature_names=feature_names)
                            preds.append(batch_pred)
                        y_pred_valid = np.concatenate(preds)
                        y_pred_flat[valid_mask] = y_pred_valid

                    y_pred_tile = y_pred_flat.reshape(h, w)
                    y_pred_final[y:y+h, x:x+w] = y_pred_tile

                    pbar.update(1)

    return y_pred_final, meta

def save_array_as_tif(array, reference_meta, save_dir, model_type):
    """
    Saves a 2D prediction result as a GeoTIFF image.

    Parameters:
    ----------
    array : ndarray
        2D array of prediction values.
    reference_meta : dict
        Metadata from a reference image to maintain spatial alignment.
    save_dir : str
        Directory to save the output.
    model_type : str
        Type of the model (used for filename).

    Returns:
    ----------
    full_path : str
        Full path to the saved GeoTIFF file.
    """
    os.makedirs(save_dir, exist_ok=True)
    tif_path = os.path.join(save_dir, f'{model_type}_prediction_results.tif')
    meta = reference_meta.copy()
    meta.update({
        'count': 1,
        'dtype': 'float32'
    })
    with rasterio.open(tif_path, 'w', **meta) as dst:
        dst.write(array.astype(np.float32), 1)

    print(f"✅ Prediction image saved at: {tif_path}")
    return tif_path


def log_and_print(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def print_memory(note='', log_file=None):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    message = f"🧠 Current memory usage [{note}]: {mem:.2f} MB"
    log_and_print(message, log_file)


def print_time(start_time, note='', log_file=None):
    elapsed = time.time() - start_time
    message = f"⏱️ Elapsed time [{note}]: {elapsed:.2f} seconds"
    log_and_print(message, log_file)


import threading
import psutil
import os
import time

def memory_monitor(interval=1, stop_event=None):
    """
    Monitors memory usage of the current process and prints it
    at regular intervals.

    Parameters:
    ----------
    interval : int
        Time interval in seconds between memory checks.
    stop_event : threading.Event
        Event used to signal stopping the monitoring loop.
    """
    process = psutil.Process(os.getpid())
    while not (stop_event and stop_event.is_set()):
        mem = process.memory_info().rss / 1024 / 1024
        print(f"[Real-time Memory Monitor] Current memory usage: {mem:.2f} MB")
        time.sleep(interval)

def Estimation_map_main(multi_tif_data_path, model_path, model_selection, bandlist, index_list, save_dir):
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=memory_monitor, args=(1, stop_event), daemon=True)
    monitor_thread.start()

    global_start = time.time()
    save_path = os.path.join(save_dir, 'Prediction_Results')
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, 'prediction_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("📒 Prediction log\n" + "=" * 40 + "\n")

    def log_and_print(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def print_time(start_time, note=''):
        elapsed = time.time() - start_time
        log_and_print(f"⏱️ Elapsed time [{note}]: {elapsed:.2f} seconds")

    try:
        log_and_print("\n🚀 Starting Estimation_map_main...")
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        log_and_print(f"🧠 Memory usage at startup: {mem:.2f} MB")

        start = time.time()
        if model_selection == 'ML':
            path = model_path[0]
        elif model_selection == 'DL':
            path = model_path[1]
        elif model_selection == 'Auto':
            path = model_path[2]
        else:
            raise ValueError(f"❌ Unsupported model format: {model_path}")
        model, model_type = load_model(path, model_selection)
        print_time(start, "Model loading")
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        log_and_print(f"🧠 Memory usage after model loading: {mem:.2f} MB")

        if not isinstance(multi_tif_data_path, list):
            multi_tif_data_path = [multi_tif_data_path]

        pre_tif_path_list = []

        for tif_path in multi_tif_data_path:
            log_and_print(f"\n📍 Processing image: {tif_path}")

            start = time.time()
            y_pred, meta = predict_from_tif_tile_based(tif_path, model, model_type, bandlist, index_list)
            print_time(start, "Image prediction")
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            log_and_print(f"🧠 Memory usage after prediction: {mem:.2f} MB")

            start = time.time()
            pre_tif_path = save_array_as_tif(y_pred, meta, save_path, model_type)
            print_time(start, "Image saving")
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            log_and_print(f"🧠 Memory usage after saving prediction image: {mem:.2f} MB")

            start = time.time()
            fig = visualize_tif.plot_percentile_clipping_hist(pre_tif_path, clip_percentile=5)
            file_name = os.path.splitext(os.path.basename(pre_tif_path))[0]
            fig_save_path = os.path.join(save_path, f"{file_name}.png")
            fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
            fig.clf()
            print_time(start, "Histogram generation and saving")
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            log_and_print(f"🧠 Memory usage after saving histogram: {mem:.2f} MB")

            pre_tif_path_list.append(pre_tif_path)

        start = time.time()
        image_paths = sorted([os.path.join(save_path, f) for f in os.listdir(save_path) if f.lower().endswith('.png')])
        images = [Image.open(path) for path in image_paths]
        print_time(start, "Loading PIL images")
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        log_and_print(f"🧠 Final memory usage before returning results: {mem:.2f} MB")

        print_time(global_start, "Total runtime")
        log_and_print("✅ All tasks completed!")

        return images, pre_tif_path_list

    finally:
        stop_event.set()
        monitor_thread.join()

if __name__ == "__main__":

    multi_tif_data_path = r"D:\Code_Store\InversionSoftware\S3_prediction_vis\UAV_image\20250421ROI.tif"
    model_path = [
        'D:\Code_Store\InversionSoftware\S3_prediction_vis\s2\RandomForest_best_ML_model.joblib',
        'D:\Code_Store\InversionSoftware\S3_prediction_vis\s2\CNN_best_DL_model.keras',
        'D:\Code_Store\InversionSoftware\S3_prediction_vis\s2\AutoGluon'
    ]

    model_selection = 'Auto'
    save_dir = r"D:\Code_Store\InversionSoftware\S3_prediction_vis"
    bandlist = ['green', 'nir', 'red', 'rededge']

    index_list = ["RED", "ARI", "CVI", "MCARI1", "MCARI4", "MNLI", "NDGI", "NNIR", "PSRI2", "REDVI", "RI", "VI700"]

    images, pre_tif_path_list = Estimation_map_main(multi_tif_data_path, model_path, model_selection, bandlist, index_list, save_dir)
