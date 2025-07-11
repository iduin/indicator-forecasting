import os
import json
import pandas as pd
from tqdm import tqdm

def test_labels(labels_path, data_folder, indics):
    """
    Quick check: verify labels by comparing CSV last and t-15 rows for specified indicators.

    Args:
        labels_path (str): Path to JSON file with labels.
        data_folder (str): Folder containing CSV files.
        indics (list[str]): List of indicator column names to check.
    """
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)

    for png_path in tqdm(labels_dict, desc="Testing labels"):
        csv_path = os.path.join(data_folder, png_path.split('.')[0] + '.csv')
        if not os.path.exists(csv_path):
            tqdm.write(f"Missing CSV for {png_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        for i, indic in enumerate(indics):
            # Compare last row vs row at -16 index for indicator
            # Convert boolean to int for comparison with label
            computed_label = int(df[indic].iloc[-1] > df[indic].iloc[-16])
            if computed_label != labels_dict[png_path][i]:
                tqdm.write(f"Error at {png_path}, indicator: {indic}")

def test_labels_strong(labels_path, data_folder, raw_data_path, indics):
    """
    Stronger validation: compare labels against raw Excel data based on dates.

    Args:
        labels_path (str): Path to JSON file with labels.
        data_folder (str): Folder containing CSV files.
        raw_data_path (str): Path to Excel file with raw data sheets.
        indics (list[str]): List of indicator column names to check.
    """
    raw_data = pd.ExcelFile(raw_data_path)
    sheets_dict = {}
    
    # Load all sheets once
    for sheet_name in raw_data.sheet_names:
        sheets_dict[sheet_name] = raw_data.parse(sheet_name, index_col='Unnamed: 0')

    with open(labels_path, "r") as f:
        labels_dict = json.load(f)

    for png_path in tqdm(labels_dict, desc="Strong label testing"):
        base_name = png_path.split('.')[0]
        csv_path = os.path.join(data_folder, base_name + '.csv')

        if not os.path.exists(csv_path):
            tqdm.write(f"Missing CSV for {png_path}")
            continue
        
        df_csv = pd.read_csv(csv_path)
        df_csv['Date'] = pd.to_datetime(df_csv['Date'])
        pred_date = df_csv['Date'].max()

        sheet_key = base_name.rsplit('_', 1)[0]  # Get sheet name key from filename

        if sheet_key not in sheets_dict:
            tqdm.write(f"Sheet {sheet_key} not found for {png_path}")
            continue
        
        df_raw = sheets_dict[sheet_key]

        # If 'Date' column missing, create a dummy date range for alignment
        if 'Date' not in df_raw.columns:
            df_raw['Date'] = pd.date_range(start='2023-01-01', periods=len(df_raw), freq='D')

        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        # Locate rows corresponding to prediction date
        valt_15 = df_raw[df_raw['Date'] == pred_date]
        if valt_15.empty:
            tqdm.write(f"Prediction date {pred_date} not found in raw data for {png_path}")
            continue

        pred_date_idx = valt_15.index[0]

        if pred_date_idx < 15:
            tqdm.write(f"Not enough history before prediction date in raw data for {png_path}")
            continue

        valt = df_raw.iloc[pred_date_idx - 15]

        for i, indic in enumerate(indics):
            val_15_value = valt_15.iloc[0][indic]
            val_t_value = valt[indic]

            computed_label = int(val_15_value > val_t_value)
            if computed_label != labels_dict[png_path][i]:
                tqdm.write(f"Error at {png_path}, indicator: {indic}")


if __name__ == "__main__":
    DATA_DIR = "data"

    LABEL_DIR = "labels"

    labels_paths = [os.path.join(LABEL_DIR,'train_synth_scaled_labels.json'),os.path.join(LABEL_DIR,'test_synth_scaled_labels.json')]

    data_folders = [os.path.join(DATA_DIR,'train_synth_scaled'), os.path.join(DATA_DIR,'test_synth_scaled')]

    raw_data_path = 'Base_Test_2500pts avec Synthétiques.xlsx'

    indics = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']

    for label_path, data_folder in zip(labels_paths, data_folders) :
        test_labels_strong(label_path, data_folder, raw_data_path, indics)