import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    f1_score
)
from model_evaluation.inference import inference
import torch
from tqdm import tqdm
from models_architectures.vit import load_model_VIT
from models_architectures.resnet import load_model_resnet_18
from models_architectures.squeezenet import load_model_squeezenet
from data_processing.dataset import get_dataloader, get_train_val_loaders
import os
import random
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

sns.set(style="whitegrid")

def plot_full_evaluation_dashboard(y_true, y_pred, y_prob, sample_lengths, label_names, save_path=None):
    # === 1. Collect Per-label Metrics ===
    metrics = {"Label": [], "Metric": [], "Score": []}
    for i, label in enumerate(label_names):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = np.nan

        metrics["Label"] += [label] * 5
        metrics["Metric"] += ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        metrics["Score"] += [acc, precision, recall, f1, auc]

    df = pd.DataFrame(metrics)

    # === 2. Compute Macro and Micro Metrics ===
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    summary_text = (
        f"Macro Avg:\n  Precision = {macro_p:.3f}\n  Recall = {macro_r:.3f}\n  F1 = {macro_f1:.3f}\n\n"
        f"Micro Avg:\n  Precision = {micro_p:.3f}\n  Recall = {micro_r:.3f}\n  F1 = {micro_f1:.3f}"
    )

    # === 3. Create Unified Figure ===
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1])

    # === Barplot of Metrics ===
    ax_bar = fig.add_subplot(grid[0, :3])
    sns.barplot(data=df, x="Label", y="Score", hue="Metric", palette="Set2", ax=ax_bar)
    ax_bar.set_title("Per-label Classification Metrics", fontsize=16)
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(axis='x', rotation=45)
    ax_bar.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add value labels on bars
    for p in ax_bar.patches:
        height = p.get_height()
        if not (np.isnan(height) or height == 0):
            ax_bar.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, color='black', rotation=90)

    ax_bar.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc='upper left')

    # === Summary Box ===
    ax_bar.text(5.5, 0.1, summary_text, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))

    # === Confusion Matrices ===
    for i in range(6):
        row = 1 + i // 3
        col = i % 3
        ax_cm = fig.add_subplot(grid[row, col])
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
        ax_cm.set_title(f"{label_names[i]}")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

    # === Final Layout ===
    plt.suptitle("Multilabel Classification Performance Summary", fontsize=18, y=0.98)
    plt.tight_layout()#rect=[0, 0, 1, 0.96])    
    
        # === 4. Performance vs. 'len' by Label ===
    ax_curve = fig.add_subplot(grid[2, :])  # Use full width of last row for clarity

    df_perf = pd.DataFrame({
        "len": np.array(sample_lengths).flatten()
    })

    bins = [50, 100, 150, 200, 250, 300]
    df_perf["len_bin"] = pd.cut(df_perf["len"], bins=bins, right=False)

    f1_by_bin_label = []

    for i, label in enumerate(label_names):
        # Build temp df for current label
        temp_df = pd.DataFrame({
            "len_bin": df_perf["len_bin"],
            "y_true": y_true[:, i],
            "y_pred": y_pred[:, i]
        })

        grouped = temp_df.groupby("len_bin")
        for bin_name, group in grouped:
            if len(group) == 0:
                continue
            f1 = f1_score(group["y_true"], group["y_pred"], zero_division=0)
            f1_by_bin_label.append({"len_bin": bin_name, "Label": label, "F1": f1})

    df_f1_curve = pd.DataFrame(f1_by_bin_label)
    df_f1_curve["len_bin"] = df_f1_curve["len_bin"].astype(str)

    sns.lineplot(data=df_f1_curve, x="len_bin", y="F1", hue="Label", marker="o", ax=ax_curve)
    ax_curve.set_title("F1-Score vs. 'len' bin for Each Indicator", fontsize=14)
    ax_curve.set_ylabel("F1-Score")
    ax_curve.set_xlabel("'len' bin")
    ax_curve.set_ylim(0, 1)
    ax_curve.tick_params(axis='x', rotation=45)

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

def plot_full_evaluation_dashboard(y_true, y_pred, y_prob, sample_lengths, label_names, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, confusion_matrix, f1_score
    )

    sns.set(style="whitegrid")

    # === 1. Per-label Metrics ===
    metrics = {"Label": [], "Metric": [], "Score": []}
    for i, label in enumerate(label_names):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = np.nan

        metrics["Label"] += [label] * 5
        metrics["Metric"] += ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        metrics["Score"] += [acc, precision, recall, f1, auc]

    df = pd.DataFrame(metrics)

    # === 2. Macro & Micro Metrics ===
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    summary_text = (
        f"Macro Avg:\nPrecision = {macro_p:.3f} | Recall = {macro_r:.3f} | F1 = {macro_f1:.3f}\n"
        f"Micro Avg:\nPrecision = {micro_p:.3f} | Recall = {micro_r:.3f} | F1 = {micro_f1:.3f}"
    )

    # === 3. Create Figure Layout ===
    fig = plt.figure(figsize=(20, 18))
    grid = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1.2], hspace=0.8)

    # === A. Barplot ===
    ax_bar = fig.add_subplot(grid[0, :])
    sns.barplot(data=df, x="Label", y="Score", hue="Metric", palette="Set2", ax=ax_bar)
    ax_bar.set_title("Per-label Classification Metrics", fontsize=18, weight="bold")
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(axis='x', rotation=0)
    ax_bar.set_xlabel("Indicator")
    ax_bar.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Value labels
    for p in ax_bar.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax_bar.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=8, color='black', rotation=90)

    # Summary Box
    ax_bar.text(len(label_names) + 0.5, 0.3, summary_text, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray"))

    # === B. Confusion Matrices ===
    for i in range(6):
        row = 1 + i // 3
        col = i % 3
        ax_cm = fig.add_subplot(grid[1, i])
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], normalize= 'pred')
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax_cm,
                    annot_kws={"fontsize": 10}, vmin=0, vmax=1)
        ax_cm.set_title(label_names[i], fontsize=13)
        ax_cm.set_xlabel("Predicted", fontsize=10)
        ax_cm.set_ylabel("Actual", fontsize=10)

    # === C. F1 Curve by 'len' ===
    ax_curve = fig.add_subplot(grid[2, :])
    df_perf = pd.DataFrame({
        "len": np.array(sample_lengths).flatten()
    })
    bins = np.linspace(50,300,26)
    df_perf["len_bin"] = pd.cut(df_perf["len"], bins=bins, right=False)

    f1_by_bin_label = []
    for i, label in enumerate(label_names):
        temp_df = pd.DataFrame({
            "len_bin": df_perf["len_bin"],
            "y_true": y_true[:, i],
            "y_pred": y_pred[:, i]
        })
        grouped = temp_df.groupby("len_bin", observed=False)
        for bin_name, group in grouped:
            if len(group) == 0:
                continue
            f1 = accuracy_score(group["y_true"], group["y_pred"])
            f1_by_bin_label.append({"len_bin": str(bin_name), "Label": label, "Accuracy": f1})

    df_f1_curve = pd.DataFrame(f1_by_bin_label)
    sns.lineplot(data=df_f1_curve, x="len_bin", y="Accuracy", hue="Label",  ax=ax_curve)
    ax_curve.set_title("Accuracy vs. Sample Size by Label", fontsize=16)
    ax_curve.set_ylabel("Accuracy")
    ax_curve.set_xlabel("Sample Size")
    ax_curve.set_ylim(0, 1)
    ax_curve.tick_params(axis='x', rotation=45)

    # === Final Layout & Save ===
    plt.suptitle("Multilabel Classification Performance Summary", fontsize=22, y=0.99, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()



if __name__ == "__main__" :

    load_dotenv()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.getenv("DATA_DIR")

    TYPE_OF_DATA = "scaled"
    
    if TYPE_OF_DATA == "" :
        str_to_add = ""
    else :
        str_to_add = "_" + TYPE_OF_DATA

    TRAIN_DIR = os.path.join(DATA_DIR, "train" + str_to_add + "/")
    TEST_DIR = os.path.join(DATA_DIR, "test" + str_to_add + "/")

    LABELS_DIR = "labels"
    TRAIN_LABEL_FILE = "train"+ str_to_add +"_labels.json"
    TEST_LABEL_FILE = "test"+ str_to_add +"_labels.json"
    TRAIN_LABEL_PATH = os.path.join(LABELS_DIR, TRAIN_LABEL_FILE)
    TEST_LABEL_PATH = os.path.join(LABELS_DIR, TEST_LABEL_FILE)

    BATCH_SIZE = 32

    INDICS = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']    

    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    IMAGE_SIZE = 256
    PATCH_SIZE = 6
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    PROJECTION_DIM = 64
    NUM_HEADS = 4
    TRANSFORMER_LAYERS = 8
    MLP_HEAD_UNITS = [2048, 1024]


    MODEL_DIR = 'model'
    MODEL_NAME = 'squeezenet_' + str(IMAGE_SIZE) + str_to_add +  "_final"
    MODEL_FILE = MODEL_NAME + '.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    PERF_DIR = 'performances'
    PERF_FILE = MODEL_NAME + '.png'
    PERF_PATH = os.path.join(PERF_DIR, PERF_FILE)


    train_loader, train_dataset, test_loader, test_dataset = get_train_val_loaders(train_image_dir = TRAIN_DIR, train_labels_path = TRAIN_LABEL_PATH, test_image_dir = TEST_DIR, test_labels_path = TEST_LABEL_PATH, train_batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, adapt_scaling = True)


    

    model = load_model_squeezenet(MODEL_PATH, DEVICE)
    all_labels, all_preds, all_probs, _, all_lens, _ = inference (model, test_loader, INDICS, DEVICE)

    print(all_probs)

    plot_full_evaluation_dashboard(all_labels, all_preds, all_probs, all_lens, INDICS, save_path = PERF_PATH)