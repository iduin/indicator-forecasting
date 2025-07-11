# Create_graph_multi.py reads a excel file and creates graphs for the data in the file
# The graphs are saved in the same directory as the excel file
# The graphs are saved in the format of .png

import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import pdb
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_processing.preprocessing import ECDFScaler
from dotenv import load_dotenv
from general_utils import load_json_list


def plot_rgb(data, names, pixel_size=256, dpi=100, scaler = None):
    """
    Plot groups of 3 time series as RGB channels and save an exact pixel-sized image.

    Each group of three series is plotted in red, green, and blue respectively,
    stacked vertically in the figure. The figure size is controlled via pixel size and DPI,
    ensuring no interpolation or buffer artifacts.

    Args:
        data (pd.DataFrame or np.ndarray): Time series data with columns corresponding to names.
        names (list of str): List of column names in `data` to plot.
        pixel_size (int, optional): Size in pixels for width and height of the output figure. Default is 256.
        dpi (int, optional): Dots per inch for figure resolution. Default is 100.
        scaler (object with .transform method, optional): Optional scaler to apply to data before plotting.

    Returns:
        matplotlib.figure.Figure: The created matplotlib figure object.
    """
    plt.style.use('dark_background')

    if scaler is not None :
        data = scaler.transform(data)

    num_groups = int(np.ceil(len(names) / 3))

    # Taille en pouces = pixels / dpi
    fig_height_inches = pixel_size / dpi
    fig = plt.figure(figsize=(fig_height_inches, fig_height_inches), dpi=dpi)

    # Positionnement manuel des subplots
    for i in range(num_groups):
        ax = fig.add_axes([0, 1 - (i + 1)/num_groups, 1, 1/num_groups])  # [left, bottom, width, height]
        group_names = names[i*3:(i+1)*3]
        colors = ['red', 'green', 'blue']

        for j, name in enumerate(group_names):
            if scaler is None:
                local_scaler = MinMaxScaler()
                scaled = local_scaler.fit_transform(data[[name]])
            else:
                scaled = data[[name]]
            ax.plot(scaled, color=colors[j], linewidth=1)

        ax.axis('off')
        if scaler is not None:
            ax.set_ylim(0, 1)
    
    return fig

def draw_one(data, min_size, max_size, pred, graph_size):
    """
    Extract a random sub-series of data for drawing.

    Selects a contiguous segment of the data with random length between `min_size` and `max_size`,
    starting at a random valid position. This segment is intended for prediction tasks.

    Args:
        data (pd.DataFrame): DataFrame containing time series data.
        min_size (int): Minimum length of the extracted segment.
        max_size (int): Maximum length of the extracted segment.
        pred (int): Percentage of points reserved for prediction (currently unused in this function).
        graph_size (int): Target size for graphing (currently unused in this function).

    Returns:
        pd.DataFrame: Extracted segment of the input data.
    """
    # Draw a size with an uniform distribution between min_size and max_size
    size = np.random.randint(min_size, max_size)
    # Get a random start index between 0 and the size of the values - size
    start = np.random.randint(0, data.shape[0] - size + 1)
    # Get the data to draw
    new_data = data.iloc[start:start + size]
    return new_data
   

def create_graphs(file_path, base_folder, sheets, nb_graphs_per_thousand = 300, min_size = 50, max_size = 300, pred = 15, graph_size = 256, replace = False, test = False, scaler = None, indics = []):
    """
    Generate time series graphs from Excel sheets and save as images and CSV files.

    Reads specified sheets from an Excel file, extracts specified indicators,
    optionally cleans the output directory, and creates multiple cropped and scaled
    time series graphs saved as PNG images and corresponding CSV files.

    Args:
        file_path (str): Path to the input Excel file.
        base_folder (str): Directory where output files (images and CSVs) will be saved.
        sheets (list of str): List of sheet names to process from the Excel file.
        nb_graphs_per_thousand (int, optional): Number of graphs to generate per 1000 data points. Default 300.
        min_size (int, optional): Minimum size of a cropped time series segment. Default 50.
        max_size (int, optional): Maximum size of a cropped time series segment. Default 300.
        pred (int, optional): Percentage of points reserved for prediction at the end of each segment. Default 15.
        graph_size (int, optional): Size (in pixels) of output graph images. Default 256.
        replace (bool, optional): If True, clears existing files in the output directory before processing. Default False.
        test (bool, optional): If True, uses test mode (different indexing logic). Default False.
        scaler (object with .transform method, optional): Optional scaler to normalize data before plotting.
        indics (list of str, optional): List of indicator column names to extract from sheets.

    Returns:
        None
    """
    # Read the excel file
    data = pd.ExcelFile(file_path)
    # get all sheet names

    if replace :
        for filename in os.listdir(base_folder):
            file_path = os.path.join(base_folder, filename)
            
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                tqdm.write(f'Failed to delete {file_path}. Reason: {e}')

    # Loop on all the sheets
    for sheet_name in sheets  : #sheet_names[:2]:
        sheet = data.parse(sheet_name)
        tqdm.write(f'Processing sheet {sheet_name}')
        # if the sheet is empty, skip it
        if sheet.empty:
            continue
        
        if 'Date' not in sheet.columns:
            # Create a fake 'Date' column (e.g., today or a date range)
            sheet['Date'] = pd.date_range(start='2023-01-01 00:00:00', periods=len(sheet), freq='D')
        sheet['Date'] = pd.to_datetime(sheet['Date'])
        
        value_wished = ['Date'] + indics # , 'MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']
        value_names = [x for x in value_wished if x in sheet.columns]

        mask = sheet.map(lambda x: pd.isna(x) or x == 0).any(axis=1)
        # Find the index of the first row where NOT any value is NaN or 0
        #lvalues = sheet[value_names].iloc[60:].reset_index(drop=True)
        valid_mask = ~mask
        if valid_mask.any():
            first_valid_idx = mask[~mask].index[0]
        else:
            first_valid_idx = 0
        lvalues = sheet[value_names].loc[first_valid_idx:].reset_index(drop=True)
        lvalues.replace("#DIV/0!", np.nan, inplace=True)
        lvalues.interpolate(method='linear', inplace=True)

        not_found_values = list(set(value_wished) - set(sheet.columns))
        for value_name in not_found_values : 
            tqdm.write(f'Value {value_name} not found in sheet {sheet_name}')

        # get the size of the values
        size = lvalues['Date'].shape[0]
        # Compute the number of graphs to create
        nb_graphs = np.round(size * nb_graphs_per_thousand // 1000)
        # Loop

        if test :
            nb_graphs = lvalues.shape[0] - max_size
        
        for i in tqdm(range(nb_graphs), desc=f"{sheet_name}", leave=True, dynamic_ncols=True):
            
            if test : 
                index = max_size + i
                size = np.random.randint(min_size, max_size)
                serie = lvalues[index-size:index]

            # draw one serie
            else :
                serie = draw_one(lvalues, min_size, max_size, pred, graph_size)

            serie_crop = serie[:-pred]

            fig = plot_rgb(serie_crop, [x for x in value_names if x != "Date"], scaler = scaler)            

            # Save the figure
            fig.savefig(os.path.join(base_folder, sheet_name + '_' + str(i) + '.png'), dpi=100)
            plt.close(fig)
            # in a xlsx
            serie.to_csv(os.path.join(base_folder, sheet_name + '_' + str(i) + '.csv'), index=False)

# Main
if __name__ == "__main__":
    load_dotenv ()
    #file_name = "Base_Test_2500pts v-Louis.xlsx"

    # Create graphs for the data in the excel file
    #train_sheets = ['BIIB', 'WMT', 'KO', 'CAT', 'BA', 'MMM', 'AAPL']
    #test_sheets = ['HON', 'AMZN', 'NVDA', 'BAC', 'JPM', 'XOM', 'TSLA', 'NKE']


    file_name = "Base_Test_2500pts avec Synthétiques.xlsx"
    train_synth_sheets = ['EURUSDm1', 'EURUSDm5_p1', 'EURUSDh1_p1', 'CAC-40h4_p1', 'CAC-40d1_p1', 'Ss1', 'Ss2', 'Ss3']
    test_synth_sheets = ['EURUSDm5_p2','EURUSDh1_p2', 'CAC-40h4_p2', 'CAC-40d1_p2', 'Ss4', 'Ss5']

    indics = load_json_list('INDICS')

    SCALER_DIR = os.getenv('SCALER_DIR')
    
    scaler = ECDFScaler.load(os.path.join("ecdf_scaler_synth.pkl"))

    DATA_DIR = os.getenv('DATA_DIR')

    TRAIN_DIR = os.path.join(DATA_DIR, "train_synth_scaled")
    TEST_DIR = os.path.join(DATA_DIR, "test_synth_scaled")
    
    create_graphs(file_name, TRAIN_DIR, train_synth_sheets, replace = True, indics = indics, scaler=scaler)
    create_graphs(file_name, TEST_DIR, test_synth_sheets, replace = True, test = True, indics = indics, scaler=scaler)