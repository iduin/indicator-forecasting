import torch
from torch import nn
from torchvision import models
from torchvision.models import SqueezeNet1_1_Weights
from models_architectures.training import train_network, plot_loss
import os
from data_processing.dataset import get_train_val_loaders
from data_processing.preprocessing import get_pos_weights
from dotenv import load_dotenv
from models_architectures.utils import load_model_safely

class SqueezeNet_Multilabel(nn.Module):
    """
    SqueezeNet-based model for multi-label classification.

    This model uses SqueezeNet 1.1 as a feature extractor and replaces the classifier
    with a new head suitable for predicting multiple independent labels.

    Args:
        num_classes (int): The number of output labels. Default is 6.
        weights (torchvision.models.WeightsEnum or None): Pre-trained weights to use. 
                                                          Default is SqueezeNet1_1_Weights.DEFAULT.
    """
    def __init__(self, num_classes=6, weights=SqueezeNet1_1_Weights.DEFAULT):
        super(SqueezeNet_Multilabel, self).__init__()
        self.model = models.squeezenet1_1(weights=weights)

        # Replace the classifier to output num_classes scores instead of 1000 classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
        """
        x = self.model(x)
        return x.view(x.size(0), -1)  # Flatten the output to [batch_size, num_classes]


# Set the device to GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_squeezenet(path, device=DEVICE):
    """
    Loads a pre-trained SqueezeNet multi-label classification model from disk.

    Args:
        path (str): File path to the saved model weights (.pth file).
        device (torch.device): Device on which to load the model.

    Returns:
        SqueezeNet_Multilabel: The loaded model set to evaluation mode.
    """
    model = SqueezeNet_Multilabel()
    model = model.to(device)

    # Load model state dict from the given path
    state_dict = torch.load(path, map_location=device)

    # Load safely to avoid key mismatches
    model = load_model_safely(model, state_dict)

    model.eval()  # Switch to evaluation mode
    return model


if __name__ == '__main__' :

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

    LABELS_DIR = os.getenv("LABELS_DIR")
    TRAIN_LABEL_FILE = "train"+ str_to_add +"_labels.json"
    TEST_LABEL_FILE = "test"+ str_to_add +"_labels.json"
    TRAIN_LABEL_PATH = os.path.join(LABELS_DIR, TRAIN_LABEL_FILE)
    TEST_LABEL_PATH = os.path.join(LABELS_DIR, TEST_LABEL_FILE)

    BATCH_SIZE = 32

    INDICS = ['MACD (12,26,9)', 'STOCH-R (14)', 'STOCH-RL (15,15,1)', 'RSI (14)', 'ADX (14)', 'CCI (20)']    

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0#.00001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    IMAGE_SIZE = 256


    MODEL_DIR = 'model'
    MODEL_NAME = 'squeezenet_' + str(IMAGE_SIZE) + str_to_add +  "_final"
    MODEL_FILE = MODEL_NAME + '.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    PERF_DIR = 'performances'
    LOSS_FILE = MODEL_NAME + '_loss.png'
    LOSS_PATH = os.path.join(PERF_DIR, LOSS_FILE)


    train_loader, train_dataset, val_loader, val_dataset = get_train_val_loaders(train_image_dir = TRAIN_DIR, train_labels_path = TRAIN_LABEL_PATH, test_image_dir = TEST_DIR, test_labels_path = TEST_LABEL_PATH, train_batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, adapt_scaling = True)

    pos_weights = get_pos_weights(train_loader, device = DEVICE)    

    model = SqueezeNet_Multilabel().to(DEVICE)

    optimizer = torch.optim.AdamW(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    print(MODEL_NAME)

    history = train_network(
        model=model,
        num_epochs=NUM_EPOCHS,
        optimizer=optimizer,
        loss_function=loss_function,
        trainloader=train_loader,
        validloader=val_loader,
        device=DEVICE,
        export_path= MODEL_PATH,
        min_lr = 1e-8,
        lr_decay_factor=0.25
    )

    plot_loss(history['train_loss'], history['test_loss'], MODEL_NAME, LOSS_PATH)