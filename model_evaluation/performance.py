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
from models_architectures.general import load_model_general
from data_processing.dataset import get_dataloader, get_train_val_loaders
import os
import random
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

sns.set(style="whitegrid")

def plot_full_evaluation_dashboard(y_true, y_pred, y_prob, label_names, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, confusion_matrix, f1_score, roc_curve
    )
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    # === 1. Per-label Metrics ===
    metrics = {"Label": [], "Metric": [], "Score": []}
    roc_data = []
    
    for i, label in enumerate(label_names):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            # Données pour courbe ROC
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_data.append({'label': label, 'fpr': fpr, 'tpr': tpr, 'auc': auc})
        except ValueError:
            auc = np.nan

        metrics["Label"] += [label] * 4  # Suppression de AUC du barplot
        metrics["Metric"] += ["Accuracy", "Precision", "Recall", "F1"]
        metrics["Score"] += [acc, precision, recall, f1]

    df = pd.DataFrame(metrics)

    

    # === 3. Create Figure Layout ===
    fig = plt.figure(figsize=(18, 12))
    
    # Ajustement des ratios et espacement
    gs = fig.add_gridspec(4, 6, 
                         height_ratios=[1.5, 0.1, 1.2, 1.3], 
                         width_ratios=[1, 1, 1, 1, 1, 0.3],
                         hspace=0.4, wspace=0.3,
                         left=0.05, right=0.95, top=0.93, bottom=0.07)

    # === A. Barplot ===
    ax_bar = fig.add_subplot(gs[0, :5])  # Prend les 5 premières colonnes
    
    # Création du barplot avec une palette plus lisible
    bars = sns.barplot(data=df, x="Label", y="Score", hue="Metric", 
                      palette="viridis", ax=ax_bar)
    
    #ax_bar.set_title("Per-label Classification Metrics", fontsize=16, weight="bold", pad=20)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Indicator", fontsize=12, weight="bold")
    ax_bar.set_ylabel("Score", fontsize=12, weight="bold")
    
    # Rotation des labels x pour éviter les chevauchements
    ax_bar.tick_params(axis='x', rotation=45, labelsize=10)
    ax_bar.tick_params(axis='y', labelsize=10)
    
    # Légende repositionnée
    ax_bar.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    # Value labels avec rotation pour éviter les chevauchements
    for p in ax_bar.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax_bar.annotate(f'{height:.2f}', 
                           (p.get_x() + p.get_width() / 2., height + 0.01),
                           ha='center', va='bottom', fontsize=8, 
                           color='black', rotation=90)
            

    # === B. Confusion Matrices ===
    # Réorganisation en 2 lignes de 3 matrices
    cm_positions = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    
    for i in range(min(6, len(label_names))):
        row, col = cm_positions[i]
        ax_cm = fig.add_subplot(gs[row, col])
        
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], normalize='all')
        
        # Matrice de confusion avec une meilleure colormap
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                   cbar=True, ax=ax_cm, cbar_kws={'shrink': 0.6},
                   annot_kws={"fontsize": 9}, vmin=0, vmax=1,
                   square=True)
        
        ax_cm.set_title(f"{label_names[i]}", fontsize=12, weight="bold", pad=10)
        ax_cm.set_xlabel("Predicted", fontsize=10)
        ax_cm.set_ylabel("Actual", fontsize=10)
        ax_cm.tick_params(labelsize=9)

    # === C. ROC Curves ===
    # Utilisation des colonnes restantes pour la courbe
    
    ax_roc = fig.add_subplot(gs[2:4, 3:6])
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    for i, data in enumerate(roc_data):
        if not np.isnan(data['auc']):
            ax_roc.plot(data['fpr'], data['tpr'], 
                       label=f"{data['label']} (AUC={data['auc']:.3f})",
                       color=colors[i], linewidth=2)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate', fontsize=10)
    ax_roc.set_title('ROC Curves', fontsize=12, weight="bold")
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # === D. Macro & Micro Metrics ===
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    summary_text = (
        f"Macro Avg:\nPrecision = {macro_p:.3f}\nRecall = {macro_r:.3f}\nF1 = {macro_f1:.3f}\n\n"
        f"Micro Avg:\nPrecision = {micro_p:.3f}\nRecall = {micro_r:.3f}\nF1 = {micro_f1:.3f}"
    )
    ax_summary = fig.add_subplot(gs[1, 5:])
    ax_summary.axis('off')
    
    summary_text = (
        f"MACRO AVERAGES\n"
        f"Precision: {macro_p:.3f}\n"
        f"Recall: {macro_r:.3f}\n"
        f"F1-Score: {macro_f1:.3f}\n\n"
        f"MICRO AVERAGES\n"
        f"Precision: {micro_p:.3f}\n"
        f"Recall: {micro_r:.3f}\n"
        f"F1-Score: {micro_f1:.3f}\n\n"
        f"DATASET INFO\n"
        f"Samples: {len(y_true)}\n"
        f"Labels: {len(label_names)}"
    )
    
    ax_summary.text(0.05, 0.95, summary_text, fontsize=10, va='top',
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", 
                            edgecolor="navy", alpha=0.8),
                   transform=ax_summary.transAxes)

    # === Final Layout & Save ===
    plt.suptitle("Multilabel Classification Performance Dashboard", 
                fontsize=20, y=0.97, weight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved to {save_path}")

    plt.show()


if __name__ == "__main__" :

    load_dotenv()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.getenv("DATA_DIR")

    TYPE_OF_DATA = ""
    
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
    IMAGE_SIZE = 256    #<================================================================================================================================================
    PATCH_SIZE = 6
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    PROJECTION_DIM = 64
    NUM_HEADS = 4
    TRANSFORMER_LAYERS = 8
    MLP_HEAD_UNITS = [2048, 1024]


    MODEL_DIR = 'model'
    MODEL_NAME = 'squeezenet_' + str(IMAGE_SIZE) + str_to_add +  "_final"   #<================================================================================================================================================
    MODEL_FILE = MODEL_NAME + '.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    PERF_DIR = 'performances'
    PERF_FILE = MODEL_NAME + '.png'
    PERF_PATH = os.path.join(PERF_DIR, PERF_FILE)


    train_loader, train_dataset, test_loader, test_dataset = get_train_val_loaders(train_image_dir = TRAIN_DIR, train_labels_path = TRAIN_LABEL_PATH, test_image_dir = TEST_DIR, test_labels_path = TEST_LABEL_PATH, train_batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, adapt_scaling = True)


    

    model = load_model_general(MODEL_PATH, DEVICE)
    all_labels, all_preds, all_probs, _, all_lens, _ = inference (model, test_loader, INDICS, DEVICE)

    plot_full_evaluation_dashboard(all_labels, all_preds, all_probs, INDICS, save_path = PERF_PATH)