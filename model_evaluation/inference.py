import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from models_architectures.vit import load_model_VIT
from models_architectures.resnet import load_model_resnet_18
from models_architectures.squeezenet import load_model_squeezenet
from data_processing.dataset import get_dataloader, compute_mean_std
import os
import random
from torch.utils.data import Subset, DataLoader
from datetime import datetime, time

def combine_arrays_to_df(arrays, columns=[]):
    arrays = [np.asarray(arr) for arr in arrays]
    
    # Validate same number of rows
    row_counts = [arr.shape[0] for arr in arrays]
    if len(set(row_counts)) != 1:
        raise ValueError("All arrays must have the same number of rows.")

    combined = np.hstack(arrays)

    # Create column names
    for i, arr in enumerate(arrays):
        prefix = f"arr{i}"
        for j in range(arr.shape[1]) :
            if i+j == len(columns) :
                columns.append(f"{prefix}_{j}")

    return pd.DataFrame(combined, columns=columns)

def format_datetime_column(df, column_name):
    df[column_name] = df[column_name].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    return df

def inference(model, test_loader, indics=None, DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()

    all_labels = []
    all_probs = []
    all_dates = []
    all_sheet = []
    all_lens = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            if len(batch) == 5:
                images, labels, dates, lens, file = batch
                has_labels = True
            elif len(batch) == 4:
                images, dates, lens, file = batch
                labels = None
                has_labels = False
            else:
                raise ValueError("Unexpected batch size")

            images = images.to(DEVICE)
            outputs = model(images)

            probs = outputs.cpu().numpy()

            all_probs.append(probs)

            if has_labels:
                all_labels.append(labels.cpu().numpy())

            dt_utc = pd.to_datetime(list(dates), utc=True)
            all_dates.append(", ".join(dt_utc.astype(str)))
            all_sheet.append(file[0].rsplit('_', 1)[0])
            all_lens.append(lens)
    
    

        all_probs = np.vstack(all_probs)
    all_dates = np.array(all_dates).reshape(-1, 1)
    all_sheet = np.array(all_sheet).reshape(-1, 1)
    all_lens = np.hstack(all_lens).reshape(-1, 1)

     # --- Check if sigmoid is needed ---
    if isinstance(all_probs, torch.Tensor):
        all_probs = all_probs.detach().cpu().numpy()

    if np.max(all_probs) > 1.0 or np.min(all_probs) < 0.0:
        all_probs = 1 / (1 + np.exp(-all_probs))
    
    all_preds = (all_probs > 0.5).astype(int)

    if all_labels:
        all_labels = np.vstack(all_labels)
    else:
        all_labels = None

    return all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet

    

def inference_df(model, test_loader, indics, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path= None) :
    label_columns =  ['label ' + i for i in indics]
    pred_columns =  ['pred ' + i for i in indics]
    all_labels, all_preds, _, all_dates, all_lens,  all_sheet = inference (model, test_loader, indics, DEVICE)
    
    if all_labels is not None :

        arrays = [all_dates, all_labels, all_preds, all_lens, all_sheet]
        columns = ['Date'] + label_columns + pred_columns  + ['Len', 'Sheet']
    
    else :
        arrays = [all_dates, all_preds, all_lens, all_sheet]
        columns = ['Date'] +  pred_columns  + ['Len', 'Sheet']

    df = combine_arrays_to_df(arrays, columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if output_path != None :
        df_xlsx = format_datetime_column(df, 'Date')
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            for sheet_name, group in df_xlsx.groupby('Sheet'):
                # Clean the sheet name to be Excel-safe (max 31 chars, no special chars)
                safe_sheet_name = str(sheet_name)[:31].replace('/', '_')
                group.drop(columns='Sheet').to_excel(writer, sheet_name=safe_sheet_name, index=False)

        print(f"✅ Excel file '{output_path}' saved with multiple sheets.")

    return df


def inference_merged_df(model, test_loader, indics, raw_data_path, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path= None) :
    label_columns =  ['label ' + i for i in indics]
    pred_columns =  ['pred ' + i for i in indics]
    all_labels, all_preds, _, all_dates, _,  all_sheet = inference (model, test_loader, indics, DEVICE)

    arrays = [all_dates, all_labels, all_preds, all_sheet]
    columns = ['Date'] + label_columns + pred_columns  + ['Sheet']

    df = combine_arrays_to_df(arrays, columns)

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    df = df.sort_values('Date').reset_index(drop=True)

    df_xlsx = format_datetime_column(df, 'Date')

    raw_xlsx = pd.ExcelFile(raw_data_path)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, group in df_xlsx.groupby('Sheet'):
            # Clean the sheet name to be Excel-safe (max 31 chars, no special chars)
            safe_sheet_name = str(sheet_name)[:31].replace('/', '_')
            print(safe_sheet_name)
            df_1 = group.drop(columns='Sheet')
            df_2 = pd.read_excel(raw_xlsx, safe_sheet_name)
            if 'Date' not in df_2.columns:
                # Create a fake 'Date' column (e.g., today or a date range)
                df_2['Date'] = pd.date_range(start='2023-01-01 00:00:00', periods=len(df_2), freq='D')
            df_2['Date'] = pd.to_datetime(df_2['Date'], utc=True)
            df_2 = format_datetime_column(df_2, 'Date')

            merged_df = pd.merge(df_1, df_2, on='Date', suffixes=('_file1', '_file2'), how = 'outer')
            merged_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    print(f"✅ Excel file '{output_path}' saved with multiple sheets.")

    return merged_df


if __name__ == '__main__' :

    RAW_DATA_PATH = 'Base_Test_2500pts avec Synthétiques.xlsx' #'Base_Test_2500pts v-Louis.xlsx'

    MODEL_DIR = 'model'
    MODEL_FILE = 'resnet_multilabel3_synth.pth' #'VIT.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_DIR = "data/"
    TEST_DIR = "test_synth/"
    TEST_LABEL_FILE = "labels\\test_synth_labels.json"
    OUTPUT_PATH = "output\\merged\\" + MODEL_FILE.split('.')[0] + '_S.xlsx'
    BATCH_SIZE = 1

    IMAGE_SIZE = 128

    indics = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']

    #mean, std = compute_mean_std (TRAIN_DIR, IMAGE_SIZE)

    test_loader, test_dataset = get_dataloader(TEST_DIR, TEST_LABEL_FILE, BATCH_SIZE, shuffle = False)#, img_size= IMAGE_SIZE, mean= mean, std = std)

    # Create a smaller test subset
    subset_size = 100
    assert subset_size <= len(test_dataset), "Subset size exceeds size of test dataset"

    # Random sample without replacement
    random.seed(42)
    indices = random.sample(range(len(test_dataset)), subset_size)

    # Wrap in Subset and new DataLoader
    small_test_dataset = Subset(test_dataset, indices)
    small_test_loader = DataLoader(small_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model_resnet_18(MODEL_PATH, DEVICE)

    df = inference_merged_df(model, test_loader, indics, RAW_DATA_PATH, output_path = OUTPUT_PATH)

    print(df.head())
