import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
data_dir = '.'  # Path to your main data directory
# The main directory should contain both 'positive' and 'negative' folders
train_dir = data_dir

# Parameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets
from torch.utils.data import random_split, ConcatDataset

# Load datasets separately
# Dataset will automatically identify 'positive' and 'negative' as classes
full_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'], is_valid_file=lambda x: x.endswith('.png'))

# Check class-to-index mapping
print('Class to index mapping:', full_dataset.class_to_idx)
# Dataset will automatically identify 'positive' and 'negative' as classes
full_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'], is_valid_file=lambda x: x.endswith('.png'))




# Split into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

from tqdm import tqdm

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(progress_bar):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'Batch': batch_idx + 1, 'Loss': loss.item()})
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    print(f'Epoch {epoch+1} complete. Average Loss: {epoch_loss / len(train_loader):.4f}')

print('Training complete.')
