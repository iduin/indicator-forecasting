import torch
from models_architectures.resnet import load_model_resnet_18
from models_architectures.squeezenet import load_model_squeezenet
from models_architectures.vit import load_model_VIT



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_general (path, device = DEVICE) :
    if 'resnet' in path.lower() :
        return load_model_resnet_18(path, device = device)
    elif 'squeeze' in path.lower() :
        return load_model_squeezenet(path, device = device)
    elif 'vit' in path.lower() :
        return load_model_VIT(path, device = device)
    else :
        raise ValueError(f"Unknown model type in path: '{path}'. Expected 'resnet', 'squeezenet', or 'vit'. Try directly using the appropriate model loader.")