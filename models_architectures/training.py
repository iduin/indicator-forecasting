import numpy as np
import torch
from typing import Callable
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_network(
    model: torch.nn.Module,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    patience: int = 5,
    lr_decay_factor: float = 0.1,
    min_lr: float = 1e-6,
    export_path: str = None
    ) -> dict:
    """Train the Network (Multi-label) with early stopping and LR decay."""
    print("Training Started")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay_factor, patience=2, min_lr=min_lr
    )
    
    train_loss_means = []
    val_loss_means = []

    for epoch in range(1, num_epochs + 1):
        sys.stdout.flush()
        train_loss = []
        valid_loss = []

        all_train_preds, all_train_labels = [], []
        all_valid_preds, all_valid_labels = [], []

        model.train()
        for x, y, *_ in tqdm(trainloader, desc=f"Epoch {epoch} [Train]"):
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()
            all_train_preds.append(preds.cpu().numpy())
            all_train_labels.append(y.cpu().numpy())

        model.eval()
        with torch.no_grad():
            for x, y, *_ in tqdm(validloader, desc=f"Epoch {epoch} [Valid]"):
                x, y = x.to(device), y.float().to(device)
                outputs = model(x)
                loss = loss_function(outputs, y)
                valid_loss.append(loss.item())

                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).int()
                all_valid_preds.append(preds.cpu().numpy())
                all_valid_labels.append(y.cpu().numpy())

        val_loss_mean = np.mean(valid_loss)
        train_loss_mean = np.mean(train_loss)

        train_loss_means.append(train_loss_mean)
        val_loss_means.append(val_loss_mean)

        print(
        f"Epoch {epoch}: "
        f"Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss_mean)

        # Early stopping check
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save best model
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            model.load_state_dict(best_model_state)  # Restore best model
            break
    
    if export_path is not None :
        torch.save(best_model_state, export_path)

    history = {'train_loss' : train_loss_means, 
                'test_loss' : val_loss_means}
    
    return history


def plot_loss(train_losses, val_losses, model_name=None, save_path=None):
    sns.set(style='darkgrid', context='notebook', palette='deep')
    
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_losses, label='Train Loss', marker='o')
    sns.lineplot(x=epochs, y=val_losses, label='Validation Loss', marker='s')
    
    title = 'Train vs Validation Loss per Epoch'
    if model_name:
        title += f' — {model_name}'
    
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")

    plt.show()

