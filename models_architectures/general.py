import torch
from models_architectures.resnet import load_model_resnet_18
from models_architectures.squeezenet import load_model_squeezenet
from models_architectures.vit import load_model_VIT



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_general(path, device=DEVICE):
    """
    Loads a machine learning model based on the model type inferred from the file path.

    The function checks for keywords in the path to determine which specific model loader
    function to use: ResNet-18, SqueezeNet, or Vision Transformer (ViT).

    Args:
        path (str): The file path to the saved model.
        device (str or torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Raises:
        ValueError: If the model type cannot be determined from the file path.
    """
    path_lower = path.lower()

    if 'resnet' in path_lower:
        return load_model_resnet_18(path, device=device)
    elif 'squeeze' in path_lower:
        return load_model_squeezenet(path, device=device)
    elif 'vit' in path_lower:
        return load_model_VIT(path, device=device)
    else:
        raise ValueError(
            f"Unknown model type in path: '{path}'. "
            "Expected 'resnet', 'squeezenet', or 'vit'. "
            "Try directly using the appropriate model loader."
        )
    
if __name__ == '__main__' :
    load_model_general('model/squeezenet_multilabel.pth')