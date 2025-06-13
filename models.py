import torch
import torch.nn as nn
from torchvision import models
from typing import Type


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SqueezeNetWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        return x.view(x.size(0), -1)

def load_model_squeezenet(path, device=DEVICE):
    base = models.squeezenet1_1(weights=None)
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 6, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Sigmoid()
    )
    model = SqueezeNetWrapper(base).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_model_resnet_18 (path, device = DEVICE) :
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 6),
        nn.Sigmoid()
    )
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

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
      batch_size: int,
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
      self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Forward Pass."""
      x = self.create_patch_layer(x)
      x = self.patch_embedding_layer(x)
      for transformer_layer in self.transformer_layers:
          x = transformer_layer(x)
      x = x[:, 0]  # CLS token
      x = self.mlp_block(x)
      logits = self.logits_layer(x)
      return self.sigmoid(logits)  # Raw logits for BCEWithLogitsLoss

def load_model_VIT(path, device=DEVICE, TRANSFORMER_LAYERS = 8, PROJECTION_DIM = 64, NUM_HEADS = 4, PATCH_SIZE = 6, IMAGE_SIZE = 72, MLP_HEAD_UNITS = [2048, 1024], BATCH_SIZE = 32):
    model = ViTMultiLabelClassifierModel(
        num_transformer_layers=TRANSFORMER_LAYERS,
        embed_dim=PROJECTION_DIM,
        feed_forward_dim=PROJECTION_DIM * 2,
        num_heads=NUM_HEADS,
        patch_size=PATCH_SIZE,
        num_patches=(IMAGE_SIZE // PATCH_SIZE) ** 2,
        mlp_head_units=MLP_HEAD_UNITS,
        num_classes=6,
        batch_size=BATCH_SIZE,
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model