import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from external.tcopil2025.fmpapi import parse_args, get_fmp_data
from external.tcopil2025.indicateurs import Indicator
from data_processing.create_training_data import plot_rgb
from data_processing.dataset import get_dataloader
from model_evaluation.inference import inference, combine_arrays_to_df
from models_architectures.resnet import load_model_resnet_18
from models_architectures.vit import load_model_VIT
from models_architectures.squeezenet import load_model_squeezenet
from models_architectures.general import load_model_general
from data_processing.preprocessing import ECDFScaler
from dotenv import load_dotenv
import os

from tqdm import tqdm
import shutil


def draw_data (df, csv_path, max_size = 285, min_size = 35) :
    if len(df) > min_size :
        size = np.random.randint(min_size, min(max_size, len(df)))
        img_df = df.tail(size)
        img_df.to_csv(csv_path)
        return img_df
    else :
        print("Data too small")

def aggregate_by_date_consistent(all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet, threshold=0.5):
    # Flatten all arrays as needed
    dates = all_dates.flatten()
    sheets = all_sheet.flatten()
    lens = all_lens.flatten()

    data = {
        'date': dates,
        'sheet': sheets,
        'len': lens
    }

    # Add labels if they exist
    if all_labels is not None:
        for i in range(all_labels.shape[1]):
            data[f'label_{i}'] = all_labels[:, i]
    else:
        all_labels = None

    # Add preds and probs columns
    for i in range(all_preds.shape[1]):
        data[f'pred_{i}'] = all_preds[:, i]
        data[f'prob_{i}'] = all_probs[:, i]

    df = pd.DataFrame(data)

    # Group by date and aggregate
    agg_dict = {}
    if all_labels is not None:
        for i in range(all_labels.shape[1]):
            agg_dict[f'label_{i}'] = 'mean'  # or 'first' if you want
    for i in range(all_preds.shape[1]):
        agg_dict[f'prob_{i}'] = 'mean'  # average probs
        agg_dict[f'pred_{i}'] = 'mean'  # average preds (will recalc anyway)

    # For 'sheet' and 'len', take first occurrence (or mode if you prefer)
    agg_dict['sheet'] = 'first'
    agg_dict['len'] = 'mean'

    grouped = df.groupby('date').agg(agg_dict).reset_index()

    # Recalculate preds based on averaged probs
    prob_cols = [f'prob_{i}' for i in range(all_probs.shape[1])]
    averaged_probs = grouped[prob_cols].values
    recalculated_preds = (averaged_probs > threshold).astype(int)

    # Prepare final arrays
    final_dates = grouped['date'].values.reshape(-1, 1)
    final_sheets = grouped['sheet'].values.reshape(-1, 1)
    final_lens = grouped['len'].values.reshape(-1, 1)

    if all_labels is not None:
        label_cols = [f'label_{i}' for i in range(all_labels.shape[1])]
        final_labels = grouped[label_cols].values
    else:
        final_labels = None

    return final_labels, recalculated_preds, averaged_probs, final_dates, final_lens, final_sheets


def inference_pipeline(model, n_avg = 1,  symbol = "AAPL", interval = '30min', indics = ['macd', 'stochRf', 'stochRL', 'rsi', 'adx', 'cci'], scaler = None, APIKEY='xQZFfDNtJjyxghjNX7YPW4VaZO1WzTif', temp_folder = 'temp_data', clean = True) :
    
    if clean :
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                tqdm.write(f'Failed to delete {file_path}. Reason: {e}')
    
    
    data = get_fmp_data(symbol, interval=interval, APIKEY = APIKEY)
    
    indicator = Indicator(data)

    indicator.macd()
    indicator.stochastic()
    indicator.rsi()
    indicator.adx()
    indicator.cci()
    df = indicator.df
    mask = df.applymap(lambda x: pd.isna(x) or x == 0).any(axis=1)
    # Find the index of the first row where NOT any value is NaN or 0
    first_valid_idx = mask[~mask].index[0]
    # Slice the DataFrame from that index onward
    df = df.loc[first_valid_idx:].reset_index(drop=True)

    for i in range (n_avg) :
        img_path = os.path.join(temp_folder, symbol + f"_{i}.png")
        csv_path = os.path.join(temp_folder, symbol + f"_{i}.csv")

        img_df = draw_data(df, csv_path)
        img = plot_rgb(img_df, indics, scaler= scaler)
        img.savefig(img_path, dpi=100)
        plt.close(img)
        
    loader, _  = get_dataloader(temp_folder, batch_size = 1)

    all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet = inference(model, loader)

    
    all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet = aggregate_by_date_consistent(all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet)

    pred_columns =  ['pred ' + i for i in indics]
    label_columns =  ['label ' + i for i in indics]

    if all_labels is not None :

        arrays = [all_dates, all_labels, all_preds, all_lens, all_sheet]
        columns = ['Date'] + label_columns + pred_columns  + ['Len', 'Sheet']
    
    else :
        arrays = [all_dates, all_preds, all_lens, all_sheet]
        columns = ['Date'] + pred_columns  + ['Len', 'Sheet']

    df = combine_arrays_to_df(arrays, columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if clean :
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                tqdm.write(f'Failed to delete {file_path}. Reason: {e}')

    return df


if __name__ == '__main__' :

    load_dotenv()

    SCALER_DIR = os.getenv('SCALER_DIR')
    
    scaler = ECDFScaler.load(os.path.join("ecdf_scaler.pkl"))

    MODEL_DIR = os.getenv('MODEL_DIR')
    MODEL_FILE = 'resnet_18_256_scaled_final.pth' #'VIT.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


    symbol = 'AAPL'

    model = load_model_general(MODEL_PATH) # model = load_model_squeezenet(MODEL_PATH) # model = load_model_VIT(MODEL_PATH)

    print(inference_pipeline(model, n_avg= 5, scaler = scaler))