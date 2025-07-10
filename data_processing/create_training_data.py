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
    Trace les séries par groupe de 3 en RGB, et sauvegarde une image exacte en pixels (ex: 256x256),
    en contrôlant la taille via figsize + dpi. Aucun buffer utilisé.
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
    Draw data in the serie
    :param data: the list of values (list) initial data
    :param min_size: the minimum size of the data to draw
    :param max_size: the maximum size of the data to draw
    :param pred: the number of values to predict in percentage
    :param graph_size: the size of the graph (number of points)
    :return: the drawn serie
    """
    # Draw a size with an uniform distribution between min_size and max_size
    size = np.random.randint(min_size, max_size)
    # Get a random start index between 0 and the size of the values - size
    start = np.random.randint(0, data.shape[0] - size + 1)
    # Get the data to draw
    new_data = data.iloc[start:start + size]
    '''new_data = []
    for serie in data:
        serie = serie[start:start + size]
        # Resample the data to the graph_size + Pred (%) using linear interpolation
        si = (np.round((1.0 + pred / 100) * graph_size)).astype(int)
        serie = np.interp(np.linspace(0, 1, si), np.linspace(0, 1, len(serie)), serie)
        new_data.append(serie)'''
    return new_data

######

def plot_one(data, names, size=256):
    """
    Draws a time series using matplotlib.
    
    :param data: 2D array-like (num_samples x num_series), each column is a series
    :param names: List of names for the series
    :param size: Size (in pixels) of the output image (assumed square)
    :return: A matplotlib figure object
    """
    # Set style before creating the figure
    plt.style.use('dark_background')

    # Create figure with desired size in inches (DPI = 100)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)

    # Plot each time series
    for values, name in zip(data.T, names):
        # Normalize values to [0, 1]
        if np.max(values) != np.min(values):  # Prevent division by zero
            values = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            values = np.zeros_like(values)  # flat line if all values are the same
        plt.plot(values, label=name, linewidth=1)

    # Remove axes and borders
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig
   

def create_graphs(file_path, base_folder, sheets, nb_graphs_per_thousand = 300, min_size = 50, max_size = 300, pred = 15, graph_size = 256, replace = False, test = False, scaler = None, indics = []):
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