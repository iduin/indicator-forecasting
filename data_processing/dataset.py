import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
from PIL import Image
import json
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import random
from general_utils import clean_filename_keep_ext

class TimeSeriesGraphDataset(Dataset):
    def __init__(self, image_dir, labels_dict = None, transform=None, use_one_hot_labels=True):
        self.image_dir = image_dir
        self.labels_dict = labels_dict
        if labels_dict is not None:            
            self.image_names = [f for f in os.listdir(image_dir) if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and clean_filename_keep_ext(f) in list(labels_dict.keys()))]
        else:
            self.image_names = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        csv_path = os.path.join(self.image_dir, img_name.split('.')[0] + '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = [col.capitalize() for col in df.columns]
            df['Date'] = pd.to_datetime(df['Date'])
            if self.labels_dict is not None:
                date = df['Date'].max().strftime('%Y-%m-%dT%H:%M:%S%z')
                length = len(df)
            else :
                step = df['Date'].diff().mode()[0]
                date = (df['Date'].max() + 15 * step).strftime('%Y-%m-%dT%H:%M:%S%z')
                length = len(df) + 15
        else:
            date = None
            length = 0

        if self.transform:
            image = self.transform(image)

        if self.labels_dict is not None:
            label = torch.tensor(self.labels_dict[clean_filename_keep_ext(img_name)], dtype=torch.float32)
            return image, label, date, length, img_name
        else:
            return image, date, length, img_name

def compute_mean_std(image_dir, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    means, stds = [], []

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img)
        means.append(tensor.mean(dim=(1, 2)))
        stds.append(tensor.std(dim=(1, 2)))

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)

    return mean.tolist(), std.tolist()

def get_dataloader(image_dir, labels_path = None, batch_size=32, shuffle=True, img_size=256, mean = None, std = None):
    if labels_path:
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
    else:
        labels_dict = None

    if not mean :
        mean = [0.485, 0.456, 0.406]

    if not std :
        std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = TimeSeriesGraphDataset(image_dir, labels_dict, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset

def get_train_val_loaders(train_image_dir, train_labels_path, test_image_dir, test_labels_path = None, train_batch_size=32, img_size=256, adapt_scaling = False, val_set_ratio = 0.25):

    if adapt_scaling :
        mean, std = compute_mean_std(train_image_dir, img_size= img_size)
    else :
        mean, std = None, None

    train_loader, train_set = get_dataloader(train_image_dir, labels_path = train_labels_path, batch_size=train_batch_size, shuffle=True, img_size=img_size, mean = mean, std = std)
    _, test_set = get_dataloader(test_image_dir, labels_path = test_labels_path, batch_size = 1, shuffle=False, img_size=img_size, mean = mean, std = std)

    val_set_size = int(len(train_set) * val_set_ratio)
    indices = random.sample(range(len(test_set)), val_set_size)
    val_set = Subset(test_set, indices)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    return train_loader, train_set, val_loader, val_set


def label_data (image_dir, labels_path, indics) :
    # Initialize label dictionary
    labels = {}

    # Iterate over files in the directory
    for file in tqdm(os.listdir(image_dir)):
        if file.endswith(".csv"):
            base_name = os.path.splitext(file)[0]
            csv_path = os.path.join(image_dir, file)
            png_path = os.path.join(image_dir, base_name + ".png")

            # Check if corresponding image exists
            if not os.path.exists(png_path):
                print(f"Skipping {file}: no matching PNG found.")
                continue

            # Load CSV data
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            # Ensure enough rows (need at least t+15)
            if len(df) < 16:
                print(f"Skipping {file}: not enough rows (need at least 16).")
                continue

            # Take row at t (index -16) and t+15 (index -1)
            val_t = df.iloc[-16]
            val_t15 = df.iloc[-1]

            # Compute binary label for each of the first 6 values
            label = [(1 if val_t15[i] > val_t[i] else 0) for i in indics]

            # Save to dictionary using image name
            labels[base_name + ".png"] = label

    # Save labels to JSON file
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)

    print(f"Saved labels for {len(labels)} images to {labels_path}")

def analyze_indicator_labels(data):

    """
    Analyze labels for a multi-label classification task.
    
    Args:
        data: Either a PyTorch Dataset (returning label tensors) 
              OR a NumPy array / PyTorch tensor of shape (N, 6)
    
    Returns:
        total_samples, label_matrix, indicator_counts, indicator_percentages,
        labels_per_sample, multi_label_dist, multi_label_percent
    """
    label_matrix = []

    # Case 1: Dataset
    if hasattr(data, '__getitem__') and not isinstance(data, (np.ndarray, torch.Tensor)):
        for sample in tqdm(data, desc="Extracting labels from dataset"):
            label_tensor = sample[1]  # Assumes label is second item
            label_matrix.append(label_tensor.numpy())
        label_matrix = np.stack(label_matrix)

    # Case 2: Array or tensor
    else:
        if isinstance(data, torch.Tensor):
            label_matrix = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            label_matrix = data
        else:
            raise TypeError("Unsupported type. Expected Dataset, NumPy array, or Torch tensor.")

    total_samples = label_matrix.shape[0]

    # 1. Count how often each indicator increased
    indicator_counts = label_matrix.sum(axis=0).astype(int)
    indicator_percentages = indicator_counts / total_samples * 100

    # 2. Count how many indicators increased per sample
    labels_per_sample = label_matrix.sum(axis=1)
    multi_label_dist = Counter(labels_per_sample)
    multi_label_percent = {k: v / total_samples * 100 for k, v in multi_label_dist.items()}

    return total_samples, label_matrix, indicator_counts, indicator_percentages, labels_per_sample, multi_label_dist, multi_label_percent


def plot_statistical_analysis (data, indics, name="Dataset") :
    print(f"\n📊 {name} - Statistical Label Analysis")

    total_samples, label_matrix, indicator_counts, indicator_percentages, _, multi_label_dist, multi_label_percent = analyze_indicator_labels(data)

    print(f"Total samples: {total_samples}")

    # Plotting
    plt.figure(figsize=(12, 5))

    # ---- Indicator Count Plot ----
    plt.subplot(1, 2, 1)
    bars = plt.bar(indics, indicator_counts)
    plt.title(f"{name} - Indicator ↑ Counts")
    plt.ylabel("Count")

    # Annotate with %
    for bar, percent in zip(bars, indicator_percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + total_samples * 0.01,
                 f"{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    # ---- Multi-label Distribution Plot ----
    plt.subplot(1, 2, 2)
    keys = sorted(multi_label_dist.keys())
    values = [multi_label_dist[k] for k in keys]
    bars2 = plt.bar(keys, values)
    plt.title(f"{name} - Indicators ↑ per Sample")
    plt.xlabel("Number of Indicators ↑")
    plt.ylabel("Number of Samples")

    # Annotate with %
    for bar, k in zip(bars2, keys):
        height = bar.get_height()
        percent = multi_label_percent[k]
        plt.text(bar.get_x() + bar.get_width()/2, height + total_samples * 0.01,
                 f"{percent:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    corr = np.corrcoef(label_matrix.T)
    sns.heatmap(corr, xticklabels=indics, yticklabels=indics, annot=True, cmap='coolwarm')
    plt.title(f"{name} - Label Correlation Matrix")
    plt.show()


if __name__ == "__main__" :

    DATA_DIR = "data"

    LABEL_DIR = "labels"

    labels_paths = [os.path.join(LABEL_DIR,'train_synth_scaled_labels.json'),os.path.join(LABEL_DIR,'test_synth_scaled_labels.json')]

    data_folders = [os.path.join(DATA_DIR,'train_synth_scaled'), os.path.join(DATA_DIR,'test_synth_scaled')]

    indics = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']

    for labels_path, data_folder in zip(labels_paths, data_folders) :
        label_data(data_folder, labels_path, indics)
    
    '''labels_paths = ['test_synth_labels.json']

    data_folders = ['test_synth']

    indics = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']

    for labels_path, data_folder in zip(labels_paths, data_folders) :
        label_data(data_folder, labels_path, indics)'''