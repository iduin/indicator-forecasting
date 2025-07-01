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
    """Custom PyTorch Layer to Extract Patches from Images."""

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
    """Positional Embedding Layer for Images of Patches."""

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
    """Create a Feed Forward Network for the Transformer Layer."""
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
    """Transformer Block Layer."""

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
  """ViT Model for Multi-Label Image Classification."""

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


'''def load_model_VIT (path, 
              device = DEVICE,
              num_transformer_layers=8,
              embed_dim=64,
              feed_forward_dim=64 * 2,
              num_heads=4,
              patch_size=6,
              num_patches=42,
              mlp_head_units=[2048, 1024],
              num_classes=6) :
    
    model = ViTMultiLabelClassifierModel(
        num_transformer_layers=num_transformer_layers,
        embed_dim=embed_dim,
        feed_forward_dim=feed_forward_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        num_patches=num_patches,
        mlp_head_units=mlp_head_units,
        num_classes=num_classes,
        device=device,
    )    
    model = model.to(device)
    state_dict  = torch.load(path, map_location=DEVICE)
    model = load_model_safely(model, state_dict)
    model.eval()
    return model'''


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