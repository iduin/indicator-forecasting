### Top 3 Unscaled Models Performance

| Model Name         | Accuracy Plot                      | Loss Plot                        |
|--------------------|----------------------------------|---------------------------------|
| resnet_multilabel2   | ![Model 1 Accuracy](performances/resnet_multilabel2.png)    | ![Model 1 Loss](performances/resnet_multilabel2_loss.png)   |
| VIT_128      | ![Model 2 Accuracy](performances/VIT_128.png)       | ![Model 2 Loss](performances/VIT_128_loss.png)      |
| squeezenet_multilabel| ![Model 3 Accuracy](performances/squeezenet_multilabel.png)| ![Model 3 Loss](performances/squeezenet_multilabel_loss.png) |


# 📚 Training & Model Usage Guide

This guide explains how to prepare your data, train models, and run inference using the code in this repository.

---

## 🛠️ 0. Data Creation from Excel Files

To generate training and testing images from financial time series stored in Excel files, follow the steps below.

### 📈 How It Works:

- Reads time series data from one or multiple Excel sheets.
- Extracts selected financial indicators (MACD, RSI, etc.).
- Randomly samples sequences of varying lengths.
- Plots these sequences as multi-line RGB graphs (each indicator as a color).
- Saves the graphs as `.png` images and the corresponding data as `.csv` files.

### 📄 Input:

- An Excel file (`.xlsx`) containing multiple sheets with time series data.
- Each sheet should include the relevant indicators you wish to visualize.

### 🖼️ Output:

- A folder with images (`.png`) of plotted time series.
- Optionally: CSV files of the original data slices.

Example output folder:
```
DATA_DIR/
├── train/
│   ├── EURUSDm1_0.png
│   ├── EURUSDm1_0.csv
│   └── ...
├── test/
│   ├── EURUSDm5_p2_0.png
│   └── ...
```

### ⚙️ Running the Script

1. Set the following variables in your `.env` file:

```
DATA_DIR=/path/to/output/folder
SCALER_DIR=/path/to/scaler/folder
INDICS=['indic_1', ...]
TRAIN_SHEETS=['train_sheet_1', ...]
TEST_SHEETS=['test_sheet_1', ...]
```

2. Here is a script you can run to create data:

```python
from data_processing.create_training_data import create_graphs
from data_processing.preprocessing import ECDFScaler
from general_utils import load_json_list
import os

DATA_DIR = os.getenv('DATA_DIR')
SCALER_DIR = os.getenv('SCALER_DIR')
INDICS = load_json_list("INDICS")
train_sheets = load_json_list("TRAIN_SHEETS")
test_sheets = load_json_list("TEST_SHEETS")

excel = 'your_excel_db.xlsx'

# Paths
train_dir = os.path.join(DATA_DIR, 'your_train_dir')
test_dir = os.path.join(DATA_DIR, 'your_test_dir')

# Here is a script you can run to create your data
scaler = ECDFScaler()
scaler.fit_excel_sheets(excel, sheet_names=train_sheets, names = INDICS)
scaler.save(os.path.join(SCALER_DIR,"your_scaler.pkl"))

scaler = ECDFScaler.load(os.path.join(SCALER_DIR,"your_scaler.pkl"))


# Create Training Images
create_graphs(file_name, train_dir, train_sheets, replace=True, indics=INDICS, scaler=scaler)

# Create Testing Images
create_graphs(file_name, test_dir, test_sheets, replace=True, test=True, indics=INDICS, scaler=scaler)
```

✅ You can adjust:
- Number of graphs per sheet (`nb_graphs_per_thousand`)
- Minimum/maximum time window sizes (`min_size`, `max_size`)
- Image dimensions (`graph_size`)


2. Now we have to put label on the data:

```python
from data_processing.dataset import label_data
from general_utils import load_json_list
import os

labels_paths = [os.path.join(LABEL_DIR,'your_train_labels.json'),os.path.join(LABEL_DIR,'your_test_labels.json')]

data_folders = [train_sheets, test_sheets]


for labels_path, data_folder in zip(labels_paths, data_folders) :
    label_data(data_folder, labels_path, INDICS)
```
---

👉 Once the data is created, you can proceed to model training as described in the next section.

---
