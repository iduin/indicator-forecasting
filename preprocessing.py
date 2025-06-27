import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm

class ECDFScaler:
    def __init__(self):
        self.ecdf_functions = {}

    def fit(self, df: pd.DataFrame):
        """
        Fit the ECDF scaler using the columns of the large DataFrame.
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
        Apply the fitted ECDF to transform a new DataFrame.
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
        self.fit(df)
        return self.transform(df)
    
    def fit_excel_sheets(self, excel_path, sheet_names=None, names=None):
        """
        Fit the scaler using multiple sheets from an Excel file.

        Args:
            excel_path (str): Path to the Excel file.
            sheet_names (list[str] or None): List of sheet names to read. If None, use all sheets.
            names (list[str] or None): Optional list of columns to use.
        """
        xls = pd.ExcelFile(excel_path)
        sheets = sheet_names if sheet_names is not None else xls.sheet_names

        all_data = []
        for sheet in sheets:
            df = pd.read_excel(xls, sheet_name=sheet, names=names)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)
        self.fit(combined_df)


def get_pos_weights (train_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) :
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