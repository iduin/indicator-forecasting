import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
import os
from general_utils import load_json_list

class ECDFScaler:
    """
    Empirical Cumulative Distribution Function (ECDF) scaler.
    Transforms each feature into its ECDF value based on the training data.
    """
    def __init__(self):
        self.ecdf_functions = {}

    def fit(self, df: pd.DataFrame):
        """
        Fit ECDF functions for each column in the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame to fit the scaler on.
        """
        for col in df.columns:
            x = df[col].dropna().sort_values().values
            # ECDF: unique sorted values and their cumulative probabilities
            y = np.linspace(0, 1, len(x), endpoint=True)
            # Store the interpolator for each feature
            self.ecdf_functions[col] = interp1d(
                x, y, bounds_error=False, fill_value=(0.0, 1.0)
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using the fitted ECDF functions.

        Parameters:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with ECDF values.
        """
        transformed = pd.DataFrame(index=df.index)
        for col in df.columns:
            if col in self.ecdf_functions:
                x = df[col].values
                f = self.ecdf_functions[col]
                transformed[col] = f(x)
            else:
                transformed[col] = df[col]
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
    
    def fit_excel_sheets(self, excel_path, sheet_names=None, names=None):
        """
        Fit the scaler using multiple sheets from an Excel file.

        Parameters:
            excel_path (str): Path to the Excel file.
            sheet_names (list[str] or None): Sheets to read. Reads all if None.
            names (list[str] or None): Optional list of column names for reading.
        """
        xls = pd.ExcelFile(excel_path)
        sheets = sheet_names if sheet_names is not None else xls.sheet_names

        all_data = []
        for sheet in sheets:
            df = pd.read_excel(xls, sheet_name=sheet, names=names)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        self.fit(combined_df)
    
    def save(self, filepath):
        """
        Save the scaler to disk.

        Parameters:
            filepath (str): File path to save the scaler.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load a saved scaler from disk.

        Parameters:
            filepath (str): Path to the saved scaler file.

        Returns:
            ECDFScaler: Loaded scaler instance.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def get_pos_weights (train_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) :
    """
    Compute positive weights for each class based on imbalance in the training labels.

    Args:
        train_loader (DataLoader): DataLoader that yields (inputs, labels, ...)
        device (torch.device or None): Device to put pos_weights on. 
                                      Defaults to CUDA if available, else CPU.

    Returns:
        torch.Tensor: Tensor of positive weights per class for use in loss functions.
    """
    # Accumulate labels
    all_labels = []

    for _, labels, *_ in tqdm(train_loader):  # *rest handles any extra outputs like dates
        all_labels.append(labels)

    # Concatenate into a single tensor
    all_labels = torch.cat(all_labels, dim=0)  # shape: [num_samples, num_classes]

    # Compute positive and negative counts per class
    positive_counts = all_labels.sum(dim=0)
    total_counts = all_labels.shape[0]
    neg_counts = total_counts - positive_counts

    # Compute pos_weight = N / P
    pos_weights = neg_counts / positive_counts

    # Define loss function with pos_weight
    pos_weights = pos_weights.to(torch.float32).to(device)

    return pos_weights

if __name__ == '__main__' :

    load_dotenv()

    SCALER_DIR = os.getenv('SCALER_DIR')
    INDICS = load_json_list("INDICS")
    train_sheets = load_json_list("TRAIN_SHEETS")
    train_synth_sheets = load_json_list("TRAIN_SYNTH_SHEETS")

    excel = 'Base_Test_2500pts v-Louis.xlsx'
    excel_synth = 'Base_Test_2500pts avec Synthétiques.xlsx'

    scaler = ECDFScaler()
    scaler.fit_excel_sheets(excel, sheet_names=train_sheets, names = INDICS)
    scaler.save(os.path.join(SCALER_DIR,"ecdf_scaler.pkl"))

    scaler_synth = ECDFScaler()
    scaler_synth.fit_excel_sheets(excel_synth, sheet_names=train_synth_sheets, names = INDICS)
    scaler.save(os.path.join(SCALER_DIR,"ecdf_scaler_synth.pkl"))

