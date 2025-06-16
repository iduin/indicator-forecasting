import os
import json
import pandas as pd
from tqdm import tqdm

def test_labels(labels_path, data_folder, indics) :
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
        for png_path in tqdm(labels_dict) :
            csv_path = os.path.join(data_folder, png_path.split('.')[0]+'.csv')
            df = pd.read_csv(csv_path)
            for i,indic in enumerate(indics) :
                if int(df[indic].iloc[-1] > df[indic].iloc[-16]) != labels_dict[png_path][i] :
                    tqdm.write(f'Error at {png_path}, {indic}')

def test_labels_strong (labels_path, data_folder, raw_data_path, indics) :
    raw_data = pd.ExcelFile(raw_data_path)
    sheets_dicc = {}
    for sheet_name in raw_data.sheet_names :
        sheets_dicc[sheet_name] = raw_data.parse(sheet_name, index_col= 'Unnamed: 0')
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
        for png_path in tqdm(labels_dict) :
            base_name = png_path.split('.')[0]
            csv_path = os.path.join(data_folder, base_name +'.csv')
            df_csv = pd.read_csv(csv_path)
            df_csv['Date'] = pd.to_datetime(df_csv['Date'])
            pred_date = df_csv['Date'].max()
            df_raw = sheets_dicc[base_name.rsplit('_', 1)[0]]
            if 'Date' not in df_raw.columns:
                # Create a fake 'Date' column (e.g., today or a date range)
                df_raw['Date'] = pd.date_range(start='2023-01-01', periods=len(df_raw), freq='D')
            df_raw['Date'] = pd.to_datetime(df_raw['Date'])
            valt_15 = df_raw[df_raw['Date'] == pd.to_datetime(pred_date)]
            pred_date_idx = valt_15.index
            valt = df_raw.iloc[pred_date_idx[0]-15]
            for i,indic in enumerate(indics) :
                #print(valt_15[indic], valt[indic], labels_dict[png_path][i])
                if int((valt_15[indic] > valt[indic]).iloc[0]) != labels_dict[png_path][i] :
                    tqdm.write(f'Error at {png_path}, {indic}')


if __name__ == "__main__":
    labels_paths = ['labels\\train_synth_labels.json','labels\\test_synth_labels.json']

    data_folders = ['data_synth', 'test_synth']

    raw_data_path = 'Base_Test_2500pts avec Synthétiques.xlsx'

    indics = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']

    for label_path, data_folder in zip(labels_paths, data_folders) :
        test_labels_strong(label_path, data_folder, raw_data_path, indics)