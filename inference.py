import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from models import load_model_squeezenet, load_model_resnet_18, load_model_VIT
from dataset import get_dataloader, compute_mean_std
import os
import random
from torch.utils.data import Subset, DataLoader

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

def inference (model, test_loader, indics, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")) :
    model.eval()
    # Evaluate
    all_labels = []
    all_preds = []
    all_probs = []
    all_dates = []
    all_sheet = []
    all_lens = []

    with torch.no_grad():
        for images, labels, dates, len, file in tqdm(test_loader) :
            images = images.to(DEVICE)
            outputs = model(images)

            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_dates.append(dates)
            all_sheet.append(file[0].split("_")[0])
            all_lens.append(len)

    # Stack all predictions and labels
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_dates = np.hstack(all_dates)
    all_dates = all_dates.reshape(-1,1)
    all_sheet = np.hstack(all_sheet)
    all_sheet = all_sheet.reshape(-1,1)
    all_lens = np.hstack(all_lens)
    all_lens = all_lens.reshape(-1,1)

    return all_labels, all_preds, all_probs, all_dates, all_lens, all_sheet

    

def inference_df(model, test_loader, indics, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path= None) :
    label_columns =  ['label ' + i for i in indics]
    pred_columns =  ['pred ' + i for i in indics]
    all_labels, all_preds, _, all_dates, all_lens,  all_sheet = inference (model, test_loader, indics, DEVICE)

    arrays = [all_dates, all_labels, all_preds, all_lens, all_sheet]
    columns = ['Date'] + label_columns + pred_columns  + ['Sheet']

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
    if df['Date'].dt.tz is None:
        df['Date'] = df['Date'].dt.tz_localize('UTC')
    df = df.sort_values('Date').reset_index(drop=True)

    df_xlsx = format_datetime_column(df, 'Date')

    raw_xlsx = pd.ExcelFile(raw_data_path)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, group in df_xlsx.groupby('Sheet'):
            # Clean the sheet name to be Excel-safe (max 31 chars, no special chars)
            safe_sheet_name = str(sheet_name)[:31].replace('/', '_')
            df_1 = group.drop(columns='Sheet')
            df_2 = pd.read_excel(raw_xlsx, safe_sheet_name)
            merged_df = pd.merge(df_1, df_2, on='Date', suffixes=('_file1', '_file2'), how = 'outer')
            merged_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

    print(f"✅ Excel file '{output_path}' saved with multiple sheets.")

    return merged_df


if __name__ == '__main__' :

    RAW_DATA_PATH = 'Base_Test_2500pts avec Synthétiques.xlsx' #'Base_Test_2500pts v-Louis.xlsx'

    MODEL_DIR = 'model'
    MODEL_FILE = 'resnet_multilabel2.pth' #'VIT.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_DIR = "data/"
    TEST_DIR = "test_synth/"
    TEST_LABEL_FILE = "labels\\test_synth_labels.json"
    OUTPUT_PATH = "output\\merged\\" + MODEL_FILE.split('.')[0] + '_synth.xlsx'
    BATCH_SIZE = 1

    IMAGE_SIZE = 72

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

    df = inference_merged_df(model, small_test_loader, indics, RAW_DATA_PATH, output_path = OUTPUT_PATH)

    print(df.head())
