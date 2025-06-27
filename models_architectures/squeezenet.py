import torch
from torch import nn
from torchvision import models
from torchvision.models import SqueezeNet1_1_Weights
from models_architectures.training import train_network, plot_loss
import os
from dataset import get_train_val_loaders
from preprocessing import get_pos_weights
from dotenv import load_dotenv

class SqueezeNet_Multilabel(nn.Module):
    def __init__(self, num_classes=6, weights=SqueezeNet1_1_Weights.DEFAULT):
        super(SqueezeNet_Multilabel, self).__init__()
        self.model = models.squeezenet1_1(weights=weights)
        
        # Replace the final fully connected layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_squeezenet (path, device = DEVICE) :
    model = SqueezeNet_Multilabel()    
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


if __name__ == '__main__' :

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

    LEARNING_RATE = 1e-5
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
    MODEL_NAME = 'squeezenet' + str(IMAGE_SIZE) + str_to_add +  "_final"
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