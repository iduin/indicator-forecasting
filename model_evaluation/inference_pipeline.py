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
import json
from general_utils import clean_folder, load_json_list

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


def prepare_augmented_df(data, indics=None, n_avg=5, temp_folder='temp_data', symbol = None):
    if indics is None:
        indics = ['macd', 'stochRf', 'stochRL', 'rsi', 'adx', 'cci']

    # Get data & indicators
    if not set(indics).issubset(data.columns) :
        indicator = Indicator(data)
        indicator.macd()
        indicator.stochastic()
        indicator.rsi()
        indicator.adx()
        indicator.cci()
        df_history = indicator.df
    else :
        df_history = data
    mask = df_history.map(lambda x: pd.isna(x) or x == 0).any(axis=1)
    df = df_history.loc[~mask].reset_index(drop=True)

    if not symbol :
        if 'Sheet' in df_history.columns:
            symbol = df_history['Sheet'].iloc[0]
        else :
            symbol = "SYMBOL"

    for i in range(n_avg):
        csv_path = os.path.join(temp_folder, f"{symbol}_{i}.csv")
        draw_data(df, csv_path)    

    return df_history

def infer_model_on_prepared_data(model, img_size, indics = None, scaler = None, temp_folder='temp_data', mean = None, std = None, labels_path= None):

    if indics is None:
        indics = ['macd', 'stochRf', 'stochRL', 'rsi', 'adx', 'cci']

    for file_name in os.listdir(temp_folder):
        if os.path.splitext(file_name)[-1].lower() == '.csv' :
            img_path = os.path.join(temp_folder, os.path.splitext(file_name)[0] + '.png')
            csv_path = os.path.join(temp_folder, file_name)
            img_df = pd.read_csv(csv_path)
            img = plot_rgb(img_df, indics, scaler=scaler)
            img.savefig(img_path, dpi=100)
            plt.close(img)

    loader, _ = get_dataloader(temp_folder, labels_path= labels_path, batch_size=1, img_size=img_size, mean = mean, std = std)
    inf = inference(model, loader, verbose = False)
    return inf

load_dotenv()

def extract_config (model_path, scaler_dir = os.getenv('SCALER_DIR'), model_paths_not_normalized = load_json_list("MODELS_NOT_NORM")) :
    config = {}
    model = load_model_general(model_path)
    config["model"] = model

    data_type = ""
    if 'synth' in model_path :
        data_type += "_synth"
    if 'scaled' in model_path :
        scaler = ECDFScaler.load(os.path.join(scaler_dir,"ecdf_scaler" + data_type + ".pkl"))
        data_type += "_scaled"
    else :
        scaler = None
    if os.path.basename(model_path) in model_paths_not_normalized :
        mean, std = None, None
    else :
        with open("mean_std.json", "r") as f:
            mean_std = json.load(f)
        mean, std = mean_std["train" + data_type]
    config["mean"] = mean
    config["std"] = std
    config["scaler"] = scaler

    img_size = 256
    if '128' in model_path :
        img_size = 128
    elif '72' in model_path :
        img_size = 72
    config["img_size"] = img_size

    return config


def inference_pipeline(
    model_paths,
    data,
    n_avg=5,
    indics=None,
    temp_folder="temp_data",
    clean=True,
    model_paths_not_normalized=None,
    labels_path= None,
    symbol = None
):
    if clean:
        clean_folder(temp_folder)

    if indics is None:
        indics = ['macd', 'stochRf', 'stochRL', 'rsi', 'adx', 'cci']

    if model_paths_not_normalized is None:
        model_paths_not_normalized = []

    # Generate the augmented data once (same augmented dataset used for all models)
    df_history = prepare_augmented_df(data, indics=indics, n_avg=n_avg, temp_folder=temp_folder, symbol = symbol)

    dfs = []

    for model_path in model_paths:
        config = extract_config(model_path)

        if model_path in model_paths_not_normalized:
            config["mean"] = None
            config["std"] = None

        # Run inference on augmented data, get raw preds, probs etc for each augmentation
        labels, preds, probs, dates, lens, sheet = infer_model_on_prepared_data(**config, indics=indics, labels_path= labels_path)

        # Now aggregate *within this model* over augmented data by date
        agg_labels, agg_preds, agg_probs, agg_dates, agg_lens, agg_sheet = aggregate_by_date_consistent(
            labels, preds, probs, dates, lens, sheet)

        pred_columns = ['pred ' + i for i in indics]
        label_columns = ['label ' + i for i in indics]

        if agg_labels is not None:
            arrays = [agg_dates, agg_labels, agg_preds, agg_lens, agg_sheet]
            columns = ['Date'] + label_columns + pred_columns + ['Len', 'Sheet']
        else:
            arrays = [agg_dates, agg_preds, agg_lens, agg_sheet]
            columns = ['Date'] + pred_columns + ['Len', 'Sheet']

        df = combine_arrays_to_df(arrays, columns)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Add the model name column for identification
        df['Model'] = os.path.basename(model_path)

        dfs.append(df)

    # Concatenate all models vertically — multiple lines per date, one per model
    combined_df = pd.concat(dfs, ignore_index=True)

    if clean:
        clean_folder(temp_folder)

    return combined_df, df_history



def inference_pipeline_preloaded(
    model_configs,
    data,
    n_avg=5,
    indics=None,
    temp_folder="temp_data",
    clean=True,
    model_paths_not_normalized=None, 
    labels_path = None,
    symbol = None
):
    if clean:
        clean_folder(temp_folder)

    if indics is None:
        indics = ['macd', 'stochRf', 'stochRL', 'rsi', 'adx', 'cci']

    if model_paths_not_normalized is None:
        model_paths_not_normalized = []

    # Generate the augmented data once (same augmented dataset used for all models)
    df_history = prepare_augmented_df(data, indics=indics, n_avg=n_avg, temp_folder=temp_folder, symbol = symbol)

    dfs = []

    for model_config in model_configs:

        # Run inference on augmented data, get raw preds, probs etc for each augmentation
        labels, preds, probs, dates, lens, sheet = infer_model_on_prepared_data(**model_config, indics=indics, labels_path = labels_path)

        # Now aggregate *within this model* over augmented data by date
        agg_labels, agg_preds, agg_probs, agg_dates, agg_lens, agg_sheet = aggregate_by_date_consistent(
            labels, preds, probs, dates, lens, sheet)

        pred_columns = ['pred ' + i for i in indics]
        label_columns = ['label ' + i for i in indics]

        if agg_labels is not None:
            arrays = [agg_dates, agg_labels, agg_preds, agg_lens, agg_sheet]
            columns = ['Date'] + label_columns + pred_columns + ['Len', 'Sheet']
            int_cols = label_columns + pred_columns
        else:
            arrays = [agg_dates, agg_preds, agg_lens, agg_sheet]
            columns = ['Date'] + pred_columns + ['Len', 'Sheet']
            int_cols = pred_columns

        df = combine_arrays_to_df(arrays, columns)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df[int_cols] = df[int_cols].apply(pd.to_numeric, errors='coerce').astype(int)

        # Add the model name column for identification
        df['Model'] = os.path.basename(model_config["model"].__class__.__name__)

        dfs.append(df)

    # Concatenate all models vertically — multiple lines per date, one per model
    combined_df = pd.concat(dfs, ignore_index=True)

    if clean:
        clean_folder(temp_folder)

    return combined_df, df_history




if __name__ == '__main__' :
    
    model_paths_not_normalized = load_json_list("MODELS_NOT_NORM")  # Liste des models dont les images n'ont pas été normalisés  ######################### GARDER CETTE LISTE ###########################        

    load_dotenv()

    SCALER_DIR = os.getenv('SCALER_DIR')
    
    MODEL_DIR = os.getenv('MODEL_DIR')
    model_files = [
        #'resnet_18_256_scaled_final.pth',
        'resnet_multilabel2.pth',
    ]
    model_paths = [os.path.join(MODEL_DIR, f) for f in model_files]

    symbol="AAPL"
    interval="30min"
    APIKEY=os.getenv('APIKEY')

    data = get_fmp_data(symbol = symbol, interval=interval, APIKEY=APIKEY)

    print(inference_pipeline(model_paths, data, n_avg= 1, model_paths_not_normalized = model_paths_not_normalized)[0])