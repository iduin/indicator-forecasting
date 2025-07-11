import torch
from torch import nn
from typing import Type

from models_architectures.training import train_network, plot_loss
import os
from data_processing.dataset import get_train_val_loaders
from data_processing.preprocessing import get_pos_weights
from dotenv import load_dotenv
from models_architectures.utils import load_model_safely

class CreatePatchesLayer(torch.nn.Module):
    """
    PyTorch layer to extract non-overlapping or overlapping patches from input images.

    This layer uses `torch.nn.Unfold` to convert 2D spatial image regions into flattened patch vectors,
    which can then be processed by transformer-based models or other sequence models.

    Args:
        patch_size (int): Size of each square patch (in pixels).
        strides (int): Stride between consecutive patches (in pixels).

    Forward Input:
        images (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Forward Output:
        torch.Tensor: Output tensor of shape (batch_size, num_patches, patch_dim).
    """

    def __init__(
    self,
    patch_size: int,
    strides: int,
    ) -> None:
        """Init Variables."""
        super().__init__()
        self.unfold_layer = torch.nn.Unfold(
            kernel_size=patch_size, stride=strides
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward Pass to Create Patches."""
        patched_images = self.unfold_layer(images)
        return patched_images.permute((0, 2, 1))

class PatchEmbeddingLayer(torch.nn.Module):
    """
    Embeds image patches into a dense vector space with positional information for transformer models.

    This layer projects flattened image patches into a fixed embedding dimension, adds a learnable 
    classification token, and injects positional embeddings to retain spatial information.

    Args:
        num_patches (int): Total number of patches extracted from the image.
        patch_size (int): Size of each patch (height and width).
        embed_dim (int): Dimension of the embedding space.
        device (torch.device): Device on which tensors should be allocated.

    Forward Input:
        patches (torch.Tensor): Tensor of shape (batch_size, num_patches, patch_dim).

    Forward Output:
        torch.Tensor: Tensor of shape (batch_size, num_patches + 1, embed_dim), including class token.
    """
    def __init__(
        self,
        num_patches: int,
        #batch_size: int,
        patch_size: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        """Init Function."""
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.position_emb = torch.nn.Embedding(
        num_embeddings=num_patches + 1, embedding_dim=embed_dim
        )
        self.projection_layer = torch.nn.Linear(
        patch_size * patch_size * 3, embed_dim
        )
        #self.class_parameter = torch.nn.Parameter(      torch.rand(batch_size, 1, embed_dim).to(device),      requires_grad=True,    )
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.device = device

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""

        batch_size = patches.size(0)
        positions = (
        torch.arange(start=0, end=self.num_patches + 1, step=1)
        .to(self.device)
        .unsqueeze(dim=0)
        )
        patches = self.projection_layer(patches)
        position_embeddings = self.position_emb(positions)
        encoded_patches = torch.cat(
        (self.class_token.expand(batch_size, -1, -1), patches), dim=1
        ) + position_embeddings.expand(batch_size, -1, -1)
        return encoded_patches


def create_mlp_block(
    input_features: int,
    output_features: list[int],
    activation_function: Type[torch.nn.Module],
    dropout_rate: float,
) -> torch.nn.Module:
    """
    Create a fully connected feed-forward block (MLP) with optional non-linearity and dropout.

    Commonly used in transformer models after attention layers to introduce non-linear transformations.

    Args:
        input_features (int): Size of the input features.
        output_features (list[int]): List of output feature sizes for each layer in the MLP.
        activation_function (Type[torch.nn.Module]): Activation function class to use (e.g., nn.GELU).
        dropout_rate (float): Dropout probability between layers.

    Returns:
        torch.nn.Sequential: Composed MLP block.
    """
    layer_list = []
    for idx in range(  # pylint: disable=consider-using-enumerate
        len(output_features)
    ):
        if idx == 0:
            linear_layer = torch.nn.Linear(
                in_features=input_features, out_features=output_features[idx]
            )
        else:
            linear_layer = torch.nn.Linear(
                in_features=output_features[idx - 1],
                out_features=output_features[idx],
            )
        dropout = torch.nn.Dropout(p=dropout_rate)
        layers = torch.nn.Sequential(
            linear_layer, activation_function(), dropout
        )
        layer_list.append(layers)
    return torch.nn.Sequential(*layer_list)

class TransformerBlock(torch.nn.Module):
    """
    Single transformer encoder block consisting of multi-head self-attention, 
    feed-forward network, layer normalization, and residual connections.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Dimension of the key and value vectors.
        embed_dim (int): Embedding dimension of the input and output.
        ff_dim (int): Hidden dimension of the feed-forward sub-layer.
        dropout_rate (float): Dropout probability applied after attention and feed-forward layers.

    Forward Input:
        inputs (torch.Tensor): Tensor of shape (batch_size, sequence_length, embed_dim).

    Forward Output:
        torch.Tensor: Tensor of same shape as input.
    """
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        embed_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.layer_norm_input = torch.nn.LayerNorm(
        normalized_shape=embed_dim, eps=1e-6
        )
        self.attn = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        kdim=key_dim,
        vdim=key_dim,
        batch_first=True,
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = torch.nn.LayerNorm(
        normalized_shape=embed_dim, eps=1e-6
        )
        self.layer_norm_2 = torch.nn.LayerNorm(
        normalized_shape=embed_dim, eps=1e-6
        )
        self.ffn = create_mlp_block(
        input_features=embed_dim,
        output_features=[ff_dim, embed_dim],
        activation_function=torch.nn.GELU,
        dropout_rate=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        layer_norm_inputs = self.layer_norm_input(inputs)
        attention_output, _ = self.attn(
        query=layer_norm_inputs,
        key=layer_norm_inputs,
        value=layer_norm_inputs,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.layer_norm_1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        output = self.layer_norm_2(out1 + ffn_output)
        return output

class ViTMultiLabelClassifierModel(torch.nn.Module):
  """
    Vision Transformer (ViT) model adapted for multi-label image classification.

    The model processes images by splitting them into patches, embedding them,
    applying multiple transformer encoder layers, and predicting multiple independent class probabilities.

    Args:
        num_transformer_layers (int): Number of transformer encoder blocks.
        embed_dim (int): Dimensionality of the patch embeddings.
        feed_forward_dim (int): Hidden size of the feed-forward layers in each transformer block.
        num_heads (int): Number of self-attention heads.
        patch_size (int): Size of the square image patches.
        num_patches (int): Number of patches the image is split into.
        mlp_head_units (list[int]): List of hidden units for the classification head MLP.
        num_classes (int): Number of independent labels to predict.
        device (torch.device): Device on which the model operates.

    Forward Input:
        x (torch.Tensor): Batch of images of shape (batch_size, channels, height, width).

    Forward Output:
        torch.Tensor: Logits of shape (batch_size, num_classes) for multi-label classification.
    """
  def __init__(
      self,
      num_transformer_layers: int,
      embed_dim: int,
      feed_forward_dim: int,
      num_heads: int,
      patch_size: int,
      num_patches: int,
      mlp_head_units: list[int],
      num_classes: int,
      device: torch.device,
  ) -> None:
      """Init Function."""
      super().__init__()
      self.create_patch_layer = CreatePatchesLayer(patch_size, patch_size)
      self.patch_embedding_layer = PatchEmbeddingLayer(
          num_patches,  patch_size, embed_dim, device
      )
      self.transformer_layers = torch.nn.ModuleList()
      for _ in range(num_transformer_layers):
          self.transformer_layers.append(
              TransformerBlock(
                  num_heads, embed_dim, embed_dim, feed_forward_dim
              )
          )

      self.mlp_block = create_mlp_block(
          input_features=embed_dim,
          output_features=mlp_head_units,
          activation_function=torch.nn.GELU,
          dropout_rate=0.5,
      )

      self.logits_layer = torch.nn.Linear(mlp_head_units[-1], num_classes)
      #self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Forward Pass."""
      x = self.create_patch_layer(x)
      x = self.patch_embedding_layer(x)
      for transformer_layer in self.transformer_layers:
          x = transformer_layer(x)
      x = x[:, 0]  # CLS token
      x = self.mlp_block(x)
      logits = self.logits_layer(x)
      return logits #self.sigmoid(logits)  # Raw logits for BCEWithLogitsLoss

import math

def infer_vit_config(state_dict):
    """
    Infer essential configuration parameters for a Vision Transformer model from a saved state_dict.

    Automatically determines key hyperparameters such as embedding dimension, number of patches,
    number of transformer layers, feed-forward dimension, and patch size.

    Args:
        state_dict (dict): State dictionary containing the pretrained model weights.

    Returns:
        dict: Dictionary of inferred configuration parameters.
    """
    config = {}

    # --- Position embedding ---
    pos_emb_shape = state_dict["patch_embedding_layer.position_emb.weight"].shape
    num_tokens, embed_dim = pos_emb_shape
    num_patches = num_tokens - 1  # exclude class token

    config["embed_dim"] = embed_dim
    config["num_patches"] = num_patches

    # --- Feedforward layer dim ---
    config["feed_forward_dim"] = state_dict["transformer_layers.0.ffn.0.0.weight"].shape[0]

    # --- Number of transformer layers ---
    config["num_transformer_layers"] = len({
        k.split('.')[1]
        for k in state_dict
        if k.startswith("transformer_layers.")
    })

    # --- Projection layer: infer patch size ---
    projection_shape = state_dict["patch_embedding_layer.projection_layer.weight"].shape
    proj_in_dim = projection_shape[1]  # [embed_dim, 3 * patch_size^2]
    patch_size = int(math.sqrt(proj_in_dim // 3))
    config["patch_size"] = patch_size

    # --- Estimate image size (assuming square image and patch grid) ---
    patch_grid = int(round(math.sqrt(num_patches)))
    image_size = patch_grid * patch_size

    #config["patch_grid"] = patch_grid
    #config["image_size"] = image_size

    # --- Optional: try to infer num_heads ---
    # NOTE: You could inspect attention weights, but we'll default safely
    config["num_heads"] = 4  # default or override manually

    return config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_VIT(path, device=DEVICE):
    """
    Load a Vision Transformer (ViT) model for multi-label classification from a saved checkpoint.

    The function infers model architecture from the checkpoint, restores weights,
    and returns the model ready for evaluation or further training.

    Args:
        path (str): Path to the saved PyTorch model checkpoint (.pth file).
        device (torch.device, optional): Device to map the model and weights to.

    Returns:
        ViTMultiLabelClassifierModel: Loaded ViT model with weights.
    """
    state_dict = torch.load(path, map_location=device)

    config = infer_vit_config(state_dict)

    # Fixed values or inferred if you can later
    config["patch_size"] = 6
    config["mlp_head_units"] = [2048, 1024]
    config["num_classes"] = 6
    config["device"] = device

    # Instantiate model
    model = ViTMultiLabelClassifierModel(**config).to(device)

    # Load state dict
    model = load_model_safely(model, state_dict)
    model.eval()
    return model


if __name__ == '__main__' :

    load_dotenv()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.getenv("DATA_DIR")

    TYPE_OF_DATA = "synth_scaled"
    
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
    IMAGE_SIZE = 128
    PATCH_SIZE = 6
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    PROJECTION_DIM = 64
    NUM_HEADS = 4
    TRANSFORMER_LAYERS = 8
    MLP_HEAD_UNITS = [2048, 1024]


    MODEL_DIR = 'model'
    MODEL_NAME = 'VIT_' + str(IMAGE_SIZE) + str_to_add +  "_final"
    MODEL_FILE = MODEL_NAME + '.pth'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    PERF_DIR = 'performances'
    LOSS_FILE = MODEL_NAME + '_loss.png'
    LOSS_PATH = os.path.join(PERF_DIR, LOSS_FILE)


    train_loader, train_dataset, val_loader, val_dataset = get_train_val_loaders(train_image_dir = TRAIN_DIR, train_labels_path = TRAIN_LABEL_PATH, test_image_dir = TEST_DIR, test_labels_path = TEST_LABEL_PATH, train_batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, adapt_scaling = True)

    pos_weights = get_pos_weights(train_loader, device = DEVICE)    

    model = ViTMultiLabelClassifierModel(
        num_transformer_layers=TRANSFORMER_LAYERS,
        embed_dim=PROJECTION_DIM,
        feed_forward_dim=PROJECTION_DIM * 2,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        mlp_head_units=MLP_HEAD_UNITS,
        num_classes=6,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    )
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    print(f"Training of {MODEL_PATH}")
    history = train_network(
        model=model,
        num_epochs=NUM_EPOCHS,
        optimizer=optimizer,
        loss_function=loss_function,
        trainloader=train_loader,
        validloader=val_loader,
        device=DEVICE,
        export_path= MODEL_PATH
    )

    plot_loss(history['train_loss'], history['test_loss'], MODEL_NAME, LOSS_PATH)