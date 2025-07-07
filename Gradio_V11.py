"""
Software interface
"""
import gradio as gr
from S1_preprocessing import d_data_process_main
from S2_model_training import model_train_main
from S3_prediction_vis import Estimation_map,calculate_indices_map,grid_grade_diagram
import pandas as pd
import geopandas as gpd
import ast
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import io

save_dir = ''
taget_value = ''

#*********************************************************************************#
#********************************** 组件生成  *************************************#
#*********************************************************************************#

def generate_dropdowns(tif_input):
    """
    Generate a list of dropdown components based on the number of bands
    in the input TIF file, for band-matching purposes.

    Parameters:
        tif_input (str or list[str]): Path to a single TIF file or a list of
                                      TIF file paths. Only the first file
                                      will be used to determine the number of bands.

    Returns:
        list[gr.Dropdown]: A list containing up to 10 Gradio Dropdown components.
                           The number of visible dropdowns is determined dynamically
                           by the number of bands in the image, with proper default
                           options and labels. If fewer than 10 bands are present,
                           hidden and non-interactive dropdowns will be appended
                           to ensure the returned list always has a length of 10.

    Notes:
        - If the file cannot be read or the number of bands is zero, the function
          will return 10 hidden dropdowns.
        - Each dropdown has options: ["BLUE", "GREEN", "NIR", "RED", "RedEdge", "other"].
        - The default value is set to "other". A maximum of 10 bands is supported
          for manual mapping.
    """

    if not isinstance(tif_input, list):
        tif_list = [tif_input]

        with rasterio.open(tif_list[0]) as src:
            band_count = src.count
    else:
        with rasterio.open(tif_input[0]) as src:
            band_count = src.count
    try:
        num = band_count
        if num <= 0:
            return [gr.Dropdown(visible=False, interactive=False)] * 5

        dropdowns = []
        for i in range(min(num, 10)):
            dropdowns.append(
                gr.Dropdown(
                    choices=["BLUE", "GREEN", "NIR", "RED", "RedEdge", "other"],
                    value="other",
                    label=f"band_ {i + 1}",
                    visible=True,
                    interactive=True
                )
            )
        # 补充隐藏的下拉框
        while len(dropdowns) < 10:
            dropdowns.append(gr.Dropdown(visible=False, interactive=False))
        return dropdowns
    except:
        return [gr.Dropdown(visible=False, interactive=False)] * 10

def generate_Checkbox(tif_input):
    """
    根据输入的文件列表生成复选框选项，用于用户自由选择数据集
    参数:
        tif_input: 可以是单个文件路径或文件路径列表
    返回:
        gr.update(): 更新复选框的配置字典
    """
    try:
        # 统一转为列表处理
        if tif_input is None:
            return gr.update(choices=[], visible=False)

        tif_list = [tif_input] if not isinstance(tif_input, list) else tif_input

        # 生成规范的选项名称（去除多余空格）
        dataset_names = [f"dataset_{i + 1}" for i in range(len(tif_list))]

        # 返回更新对象（可根据需要添加其他参数）
        return gr.update(
            choices=dataset_names,
            visible=True if dataset_names else False,
            interactive=True
        )

    except Exception as e:
        print(f"生成复选框出错: {str(e)}")
        return gr.update(choices=[], visible=False)

def generate_Number(file_input):
    """
    Generate checkbox options based on the provided list of input files,
    allowing users to freely select datasets.

    Parameters:
        tif_input: A single file path or a list of file paths.

    Returns:
        gr.update(): A dictionary to update the Gradio checkbox component's configuration.

    Notes:
        - If tif_input is None, the function returns an update that hides the checkbox.
        - Dataset names are generated in the format: "dataset_1", "dataset_2", etc.
        - In case of an error, the function prints an error message and returns
          an update that hides the checkbox.
    """
    if isinstance(file_input, list):
        file_path = file_input[0]
    else:
        file_path = file_input
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".shp":
        return gr.Number(visible=True, label="Please enter a sampling window size")
    try:
        # 尝试使用geopandas读取文件
        gdf = gpd.read_file(file_path)
        # 判断所有的矢量是否都为点
        if all(geom_type == 'Point' for geom_type in gdf.geom_type):
            return gr.Number(visible=True, label="Please enter a sampling window size")
        else:
            return gr.Number(visible=False)
    except Exception as e:
        print("发生错误:", e)
        return gr.Number(visible=False)


def generate_components(tif_input, file_input):
    """
    Generate all dynamic UI components: numeric input box, dropdowns, and checkboxes.

    Parameters:
        tif_input: Single TIF file path or a list of TIF file paths.
        file_input: Single file path or a list of file paths (e.g. shapefile).

    Returns:
        list:
            A list containing:
                - the numeric input component (for window size)
                - a list of dropdown components (for band matching)
                - the checkbox component (for selecting datasets)
    """
    # 1. Generate numeric input box (window size)
    textbox_update = generate_Number(file_input)

    # 2. Generate dropdowns and checkbox
    dropdowns = generate_dropdowns(tif_input)  # Returns a list of dropdown components
    checkbox_update = generate_Checkbox(tif_input)  # Returns the checkbox component

    # Combine all component updates into a single list
    return [textbox_update] + dropdowns + [checkbox_update]


#*********************************************************************************#
#**********************************  功能函数  *************************************#
#*********************************************************************************#

def STEP1_data_process_main(tif_paths, coord_paths, output_dir, target_column, window_size, *args):
    """
    Main entry point for data preprocessing, including parameter parsing,
    data processing execution, and image extraction.

    Parameters:
        tif_paths (str or list[str]):
            Path or list of paths to input TIF files.
        coord_paths (str or list[str]):
            Path or list of paths to coordinate files.
        output_dir (str):
            Path to the output directory.
        target_column (str):
            Name of the target field for prediction or regression.
        window_size (int):
            Window size for spatial sampling.
        *args:
            Contains multiple values from band selection dropdowns,
            and the last value is the result of the checkbox selections.

    Returns:
        best_features_df (pd.DataFrame):
            DataFrame containing the selected best features.
        sorted_feature_list (list[str]):
            List of filtered feature names.
        preview_images (list[PIL.Image]):
            List of visualization images.
        band_match (list[str]):
            Band matching information.
    """
    # Unpack dropdown and checkbox values
    dropdown_values = args[:-1]
    checkbox_values = args[-1]

    # Extract dataset indices (numeric parts)
    selected_dataset_indices = []
    if checkbox_values:
        for item in checkbox_values:
            try:
                selected_dataset_indices.append(
                    int(''.join(filter(str.isdigit, str(item)))) - 1
                )
            except (ValueError, TypeError):
                continue

    # Filter out None values for band matching
    band_match = [v for v in dropdown_values if v is not None]

    print("🧾 Parameter Summary:")
    print(f"- TIF files: {tif_paths}")
    print(f"- Coordinate files: {coord_paths}")
    print(f"- Output directory: {output_dir}")
    print(f"- Target field: {target_column}")
    print(f"- Sampling window size: {window_size}")
    print(f"- Band matching: {band_match}")
    print(f"- Selected dataset indices: {selected_dataset_indices}")

    # Call the main processing logic
    df_features, feature_list, _, result_dir = d_data_process_main.data_process_main(
        tif_paths,
        coord_paths,
        band_match,
        target_column,
        output_dir,
        selected_dataset_indices,
        window_size
    )

    # Read generated image files
    image_paths = sorted([
        os.path.join(result_dir, fname)
        for fname in os.listdir(result_dir)
        if fname.lower().endswith(".png")
    ])
    Image.MAX_IMAGE_PIXELS = None
    preview_images = [Image.open(path) for path in image_paths]

    return df_features, feature_list, preview_images, band_match


def model_data_upload(option):
    """
    Model Selection

    Description:
        - If the parameter 'option' equals "file_upload", it returns a visible file upload component.
        - If the parameter 'option' is not "file_upload", it returns a hidden file upload component.
    """
    if option == "file_upload":
        return gr.File(visible=True)
    else:
        return gr.File(visible=False)

def STEP2_model_train_main(option, file, best_features_df, output_dir_input):
    """
    Processes the input file and performs model training based on the provided option,
    returning training results and generated plots.

    Parameters:
        option (str): Specifies the operation mode. If it is "file_upload", updates data from the uploaded file.
        file (str): Path to the input file, which can be CSV or Excel format.
        best_features_df (DataFrame): Default best features DataFrame, or updated from the uploaded file.
        output_dir_input (str): Output directory used to save training results and plots.

    Returns:
        model_performance_df (DataFrame): DataFrame of model performance metrics.
        images (list of PIL.Image): List of images generated during model training.
        model_path (str): Path where the trained model file is saved.

    Description:
        - If the option is "file_upload", reads the uploaded file based on its extension and updates best_features_df.
        - Calls the `Training_main` function to perform model training and obtain model performance and path.
        - Searches for all PNG files in the specified directory, reads and returns these images.
    """
    # Check option and perform model training
    if option == "file_upload":
        if isinstance(file, list):
            if file[0].endswith('.csv'):
                best_features_df = pd.read_csv(file[0])
            elif file[0].endswith('.xlsx'):
                best_features_df = pd.read_excel(file[0])
        else:
            if file.endswith('.csv'):
                best_features_df = pd.read_csv(file)
            elif file.endswith('.xlsx'):
                best_features_df = pd.read_excel(file)

        model_performance_df, figures, model_path, save_path = model_train_main.Training_main(
            best_features_df, output_dir_input
        )
    else:
        model_performance_df, figures, model_path, save_path = model_train_main.Training_main(
            best_features_df, output_dir_input
        )

    # Search for all PNG files in the directory and sort them
    # Define sort order
    sort_order = ["ML", "DL", "auto", "best", "model"]

    # Get all .png files
    all_files = [
        os.path.join(save_path, f)
        for f in os.listdir(save_path)
        if f.lower().endswith('.png')
    ]

    # Custom sort function
    def custom_sort(file_path):
        filename = os.path.basename(file_path)
        # Assign a sort index based on the prefix of each filename
        for index, prefix in enumerate(sort_order):
            if filename.startswith(prefix):
                return index
        return len(sort_order)  # If no matching prefix, place at the end

    # Sort the file paths
    sorted_files = sorted(all_files, key=custom_sort)

    # Read as PIL image objects
    images = [Image.open(path) for path in sorted_files]

    return model_performance_df, images, model_path


def STEP3_Estimation_map_main(multi_tif_data_path, model_path, model_selection, bandlist, index_list, output_dir_input):
    """
    Executes prescription mapping for multi-band TIF data and returns generated images and TIF file paths.

    Parameters:
        multi_tif_data_path (str): Path to the input multi-band TIF file.
        model_path (str): Path to the model file used for analysis; needs to be converted into a list.
        model_selection (str): The selected model type to use.
        bandlist (str): List of bands to process; needs to be converted into a list.
        index_list (str): List of indices to compute; needs to be converted into a list.
        output_dir_input (str): Directory path to save results.

    Returns:
        images (list of PIL.Image): List of PIL images of all PNG files in the save directory.
        tif_paths (list of str): List of paths to all TIF files in the save directory.

    Description:
        - Converts parameters from text-box strings into lists.
        - Calls the main estimation map function to generate prescription TIF maps and saves results.
        - Searches the output directory for all PNG and TIF files, reads and records their paths.
    """

    # Convert string parameters from textbox into lists
    model_path = ast.literal_eval(model_path)
    bandlist = ast.literal_eval(bandlist)
    index_list = ast.literal_eval(index_list)

    # Generate prescription TIF maps and visualization histograms
    images, tif_paths = Estimation_map.Estimation_map_main(
        multi_tif_data_path,
        model_path,
        model_selection,
        bandlist,
        index_list,
        output_dir_input
    )

    return images, tif_paths

def STEP4_indices_map_main(multi_tif_data_path, bandlist, selected_option, output_dir_input):
    """
    Computes vegetation indices maps from the multi-band TIF data.

    Parameters:
        multi_tif_data_path (str): Path to the input multi-band TIF file.
        bandlist (str): List of band names as string; needs to be converted into a list.
        selected_option (str): Name of the selected vegetation index to compute.
        output_dir_input (str): Directory path to save results.

    Returns:
        tif_path (str): Path to the generated vegetation index TIF file.
        hist_img (PIL.Image): Histogram image of the generated index TIF.
    """

    # Ensure correct handling of parameters
    multi_tif_data_path = multi_tif_data_path[0]
    bandlist = ast.literal_eval(bandlist)
    selected_option = [selected_option]

    # Compute index map
    tif_path, hist_img = calculate_indices_map.indices_map_main(
        multi_tif_data_path,
        bandlist,
        selected_option,
        output_dir_input
    )

    return tif_path, hist_img

def STEP5_visualize_tif(tif_path, colormap='RdYlGn', clip_percentile=5):
    """
    Visualizes a GeoTIFF image as a PIL image, for web display or saving.

    Parameters:
    ----------
    tif_path : file-like object (Gradio input)
        Path to the input GeoTIFF file or uploaded file object.
    colormap : str
        Name of the Matplotlib color map to use for visualization.
    clip_percentile : float
        Percentage for clipping extreme values, e.g. 5 means clipping both the lower and upper 5%.

    Returns:
    ----------
    PIL.Image
        The PIL image object of the visualization.
    """
    with rasterio.open(tif_path.name) as src:
        data = src.read(1).astype(np.float32)

    valid = data[~np.isnan(data)]
    if valid.size == 0:
        raise ValueError("❌ No valid data available for visualization!")

    # Percentile clipping
    lower = np.percentile(valid, clip_percentile)
    upper = np.percentile(valid, 100 - clip_percentile)
    data_clipped = np.clip(data, lower, upper)

    # Visualization
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    norm = mpl.colors.Normalize(vmin=np.nanmin(data_clipped), vmax=np.nanmax(data_clipped))
    im = ax.imshow(data_clipped, cmap=colormap, norm=norm)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Prediction Value", fontsize=10)

    # Save to in-memory image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def STEP6_grid_main(tif_file, grid_width_meters, N_std, B_std, output_dir_input):
    """
    Generates grid-based prescription maps from nitrogen prediction results.

    Parameters:
        tif_file (str): Path to the input nitrogen prediction TIF file.
        grid_width_meters (float): Grid width in meters for creating grid-level maps.
        N_std (float): Standard nitrogen concentration.
        B_std (float): Standard field biomass.
        output_dir_input (str): Output directory path.

    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure of the grid map.
        save_path (str): Path to the saved PNG grid map.
    """

    fig, save_path = grid_grade_diagram.grid_main(
        tif_file,
        grid_width_meters,
        N_std,
        B_std,
        output_dir_input
    )

    return fig, save_path

#*********************************************************************************#
#********************************** UI界面代码 *************************************#
#*********************************************************************************#

def create_data_input_tab():
    with gr.Tab("Data Input"):
        with gr.Row():
            tif_input = gr.File(type="filepath", label="Upload remote sensing image files (.tif)", file_count="multiple")
            file_input = gr.File(type="filepath", label="Upload location file (.csv, .xlsx, .shp, .txt)", file_count="multiple")

        with gr.Row():
            output_dir = gr.Textbox(label="Set output directory")
            target_value = gr.Textbox(label="Set target column")

        upload_btn = gr.Button("Upload")

    return {
        "tif_input": tif_input,
        "file_input": file_input,
        "output_dir": output_dir,
        "target_value": target_value,
        "upload_btn": upload_btn
    }

def create_feature_engineering_tab():
    with gr.Tab("Feature Engineering"):
        with gr.Row("Band matching (used for vegetation index calculation)"):
            dropdowns = [gr.Dropdown(visible=False, interactive=False) for _ in range(10)]

        with gr.Row():
            window_size_input = gr.Number(visible=False)
            dataset_checkbox = gr.CheckboxGroup(label="Select dataset", visible=False)

        submit_btn = gr.Button("Submit parameter settings", variant="huggingface")

        sorted_features = gr.Textbox(label="Filtered feature list", show_copy_button=True, interactive=True, visible=False)
        band_order = gr.Textbox(label="Band order", show_copy_button=True, interactive=True, visible=False)

        with gr.Row():
            features_df = gr.Dataframe(label="Optimal feature preview", interactive=True, max_height=500,
                                       show_search='search', show_fullscreen_button=True, show_copy_button=True, visible=False)
            gallery = gr.Gallery(label="Visualization of feature screening results", columns=3, format='png')

    return {
        "dropdowns": dropdowns,
        "window_size_input": window_size_input,
        "dataset_checkbox": dataset_checkbox,
        "submit_btn": submit_btn,
        "sorted_features": sorted_features,
        "band_order": band_order,
        "features_df": features_df,
        "gallery": gallery
    }

def create_model_training_tab():
    with gr.Tab("Model Construction"):
        model_option = gr.Radio(
            choices=[("Upload modeling file", "file_upload"), ("Direct training", "normal_mode")],
            label="Modeling selection",
            value="file_upload"
        )
        file_upload = gr.File(label="Upload modeling data (.xlsx or .csv)", type="filepath", file_count="multiple")
        submit_btn = gr.Button("Submit modeling data")
        performance_df = gr.Dataframe(label="Model performance ranking", interactive=True, max_height=400,
                                      show_search='search', show_fullscreen_button=True, show_copy_button=True)
        gallery = gr.Gallery(label="Visualization of modeling results", columns=3, format='png')
        model_path = gr.Textbox(label="Model save path", show_copy_button=True)

    return {
        "model_option": model_option,
        "file_upload": file_upload,
        "submit_btn": submit_btn,
        "performance_df": performance_df,
        "gallery": gallery,
        "model_path": model_path
    }

def create_estimation_tab():
    index_options = ['ARI', 'ARVI', 'BDRVI', 'CCCI', 'CIG', 'CIRE', 'CIVE', 'CRI700', 'CVI', 'DATT', 'DVI', 'EBVI',
                     'EGVI', 'ERVI', 'EVI1', 'EVI2', 'ExG', 'ExGR', 'EXR1', 'EXR2', 'GARI', 'GDVI1', 'GDVI2', 'GEMI',
                     'GLI1', 'GLI2', 'GNDVI', 'GOSAVI', 'GRDVI', 'GRVI1', 'GRVI2', 'GSAVI', 'GWDRVI', 'MCARI',
                     'MCARI1', 'MCARI2', 'MCARI3', 'MCARI4', 'MDD', 'MEVI', 'MGRVI', 'MNDI1', 'MNDI2', 'MNDRE1',
                     'MNDRE2', 'MNDVI', 'MNLI', 'MNSI', 'MRETVI', 'MSAVI', 'MSR', 'MSR_G', 'MSR_R', 'MSR_RE', 'MSRRE',
                     'MSRREDRE', 'MTCARI', 'MTCI', 'NB', 'NDGI', 'NDRE', 'NDVI', 'NDWI', 'NG', 'NGBDI', 'NGI',
                     'NGRDI', 'NNIR', 'NNRI', 'NPCI', 'NR', 'NREI1', 'NREI2', 'NRI', 'NROV', 'OSAVI1', 'OSAVI2',
                     'PNDVI', 'PPR', 'PRI', 'PSRI1', 'PSRI2', 'PVR', 'RBI', 'RBNDVI', 'RDVI1', 'RDVI2', 'REDVI',
                     'REGDVI', 'REGNDVI', 'REGRVI', 'RENDVI', 'REOSAVI', 'RERVI', 'RESAVI', 'RESR', 'RETVI',
                     'REWDRVI', 'RGBVI', 'RGI', 'RI', 'RTVIcore', 'RVI', 'SAVI', 'SIPI1', 'SIPI2', 'SRI', 'SRPI',
                     'TCARI', 'TVI1', 'TVI2', 'VARI', 'VI700', 'VIopt', 'WBI', 'WDRVI', 'WI']

    with gr.Tab("Estimation Result"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                model_option = gr.Radio(
                    choices=[
                        ("Prediction using optimal machine learning model", "ML"),
                        ("Prediction using optimal deep learning model", "DL"),
                        ("Prediction using optimal ensemble learning model", "Auto")
                    ],
                    label="Model selection",
                    value="Auto"
                )
                multi_tif_path = gr.File(label="Upload TIF files", type="filepath", file_count="multiple")
                submit_btn = gr.Button("Generate prediction map")
                output_txt = gr.Textbox(label="Image saving path", show_copy_button=True, interactive=False)
                gallery = gr.Gallery(label="Prediction Histograms", columns=1, format="png")

            with gr.Column(scale=1, min_width=300):
                index_dropdown = gr.Dropdown(choices=index_options, label="Select vegetation index", interactive=True)
                index_tif_path = gr.File(label="Upload TIF file", type="filepath", file_count="multiple")
                index_btn = gr.Button("Generate index map")
                output_index_txt = gr.Textbox(label="Index map path", show_copy_button=True)
                histogram = gr.Image(label="Gray histogram", type="pil", format="png")

    return {
        "model_option": model_option,
        "multi_tif_path": multi_tif_path,
        "submit_btn": submit_btn,
        "output_txt": output_txt,
        "gallery": gallery,
        "index_dropdown": index_dropdown,
        "index_tif_path": index_tif_path,
        "index_btn": index_btn,
        "output_index_txt": output_index_txt,
        "histogram": histogram
    }

def create_prescription_tab():
    colormap_options = sorted([
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'RdYlGn', 'RdBu', 'coolwarm', 'bwr',
        'YlGn', 'terrain', 'nipy_spectral', 'twilight', 'hsv'
    ])

    with gr.Tab("Prescription Map Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                tif_file = gr.File(label="Upload GeoTIFF (.tif)", file_types=[".tif"])
                colormap = gr.Dropdown(choices=colormap_options, value="RdYlGn", label="Colormap")
                clip_slider = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Clip %")
                vis_btn = gr.Button("Visualize")
                output_img = gr.Image(label="Visualization", type="pil", format="png")

            with gr.Column(scale=1):
                tif_file_input = gr.File(label="Upload TIF image", file_types=[".tif", ".tiff"])
                grid_width = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Grid width (m)")
                N_std = gr.Number(label="Standard nitrogen concentration(g/kg)")
                B_std = gr.Number(label="Standard biomass(kg/ha)")
                grid_btn = gr.Button("Generate grid diagram", variant="primary")
                output_plot = gr.Plot(label="Grid plot")
                grid_output_path = gr.Textbox(label="Grid output path", show_copy_button=True)

    return {
        "tif_file": tif_file,
        "colormap": colormap,
        "clip_slider": clip_slider,
        "vis_btn": vis_btn,
        "output_img": output_img,
        "tif_file_input": tif_file_input,
        "grid_width": grid_width,
        "N_std": N_std,
        "B_std": B_std,
        "grid_btn": grid_btn,
        "output_plot": output_plot,
        "grid_output_path": grid_output_path
    }

with gr.Blocks(css=".container {width: 100%; margin: auto;} h1 {color: #4CAF50; text-align: center;}") as demo:
    gr.Markdown("<h1>🌾Crop Phenotypic Analysis System Based on Multispectral Remote Sensing Images</h1>")

    # Create UI of each module
    data_tab = create_data_input_tab()
    feature_tab = create_feature_engineering_tab()
    model_tab = create_model_training_tab()
    estimation_tab = create_estimation_tab()
    prescription_tab = create_prescription_tab()

    # Click event binding
    data_tab["upload_btn"].click(fn=generate_components,
                                 inputs=[data_tab["tif_input"], data_tab["file_input"]],
                                 outputs=[feature_tab["window_size_input"]] + feature_tab["dropdowns"] + [feature_tab["dataset_checkbox"]])

    feature_tab["submit_btn"].click(fn=STEP1_data_process_main,
                                    inputs=[data_tab["tif_input"], data_tab["file_input"], data_tab["output_dir"],
                                            data_tab["target_value"], feature_tab["window_size_input"],
                                            *feature_tab["dropdowns"], feature_tab["dataset_checkbox"]],
                                    outputs=[feature_tab["features_df"], feature_tab["sorted_features"],
                                             feature_tab["gallery"], feature_tab["band_order"]])

    model_tab["model_option"].change(fn=model_data_upload,
                                     inputs=model_tab["model_option"],
                                     outputs=model_tab["file_upload"])

    model_tab["submit_btn"].click(fn=STEP2_model_train_main,
                                  inputs=[model_tab["model_option"], model_tab["file_upload"],
                                          feature_tab["features_df"], data_tab["output_dir"]],
                                  outputs=[model_tab["performance_df"], model_tab["gallery"], model_tab["model_path"]])

    estimation_tab["submit_btn"].click(fn=STEP3_Estimation_map_main,
                                       inputs=[estimation_tab["multi_tif_path"], model_tab["model_path"],
                                               estimation_tab["model_option"], feature_tab["band_order"],
                                               feature_tab["sorted_features"], data_tab["output_dir"]],
                                       outputs=[estimation_tab["gallery"], estimation_tab["output_txt"]])

    estimation_tab["index_btn"].click(fn=STEP4_indices_map_main,
                                      inputs=[estimation_tab["index_tif_path"], feature_tab["band_order"],
                                              estimation_tab["index_dropdown"], data_tab["output_dir"]],
                                      outputs=[estimation_tab["output_index_txt"], estimation_tab["histogram"]])

    prescription_tab["vis_btn"].click(fn=STEP5_visualize_tif,
                                      inputs=[prescription_tab["tif_file"], prescription_tab["colormap"],
                                              prescription_tab["clip_slider"]],
                                      outputs=prescription_tab["output_img"])

    prescription_tab["grid_btn"].click(fn=STEP6_grid_main,
                                       inputs=[prescription_tab["tif_file_input"], prescription_tab["grid_width"],
                                               prescription_tab["N_std"], prescription_tab["B_std"], data_tab["output_dir"]],
                                       outputs=[prescription_tab["output_plot"], prescription_tab["grid_output_path"]])

if __name__ == "__main__":
    demo.launch()

