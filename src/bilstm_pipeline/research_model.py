"""CNN/ViT + BiLSTM/Transformer + attention model for violence detection. / Modelo CNN/ViT + BiLSTM/Transformer + atenção para detecção de violência."""

from collections.abc import Sequence
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_V2_S_Weights,
    Inception_V3_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    VisionTransformer,
    convnext_tiny,
    densenet121,
    efficientnet_v2_s,
    inception_v3,
    mobilenet_v2,
    mobilenet_v3_large,
    swin_t,
    vit_b_16,
)


class _ViTBackbone(nn.Module):
    """Wrap ViT-B/16 to accept arbitrary frame sizes and output a 1x1 spatial map. / Adapta o ViT-B/16 para qualquer resolução de quadro e saída espacial 1x1."""

    def __init__(self, net: VisionTransformer):
        super().__init__()
        net.heads = nn.Identity()
        self.net = net

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize to 224x224 when needed and extract frame embeddings. / Redimensiona para 224x224 quando necessário e extrai embeddings por quadro."""
        if frames.shape[-2:] != (224, 224):
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        return self.net(frames).unsqueeze(-1).unsqueeze(-1)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Instantiate the selected CNN or Vision Transformer feature extractor. / Instancia o extrator CNN ou Vision Transformer selecionado."""
    if name == "mobilenet_v2":
        net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        return net.features, int(net.last_channel)
    if name == "mobilenet_v3_large":
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        return net.features, int(net.classifier[0].in_features)
    if name == "densenet121":
        net = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(*net.features.children(), nn.ReLU(inplace=True)), int(net.classifier.in_features)
    if name == "inception_v3":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            net = inception_v3(
                weights=Inception_V3_Weights.DEFAULT if pretrained else None,
                aux_logits=pretrained,
                init_weights=False,
            )
        features = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            net.maxpool1, net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            net.maxpool2, net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
        )
        return features, int(net.fc.in_features)
    if name == "efficientnet_v2_s":
        net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        return net.features, int(net.classifier[1].in_features)
    if name == "convnext_tiny":
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        return net.features, int(net.classifier[2].in_features)
    if name == "swin_t":
        net = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(net.features, net.norm, net.permute), int(net.head.in_features)
    if name == "vit_b_16":
        net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        return _ViTBackbone(net), int(net.hidden_dim)
    raise ValueError(f"Unsupported backbone: {name}.")


def _configure_trainable_layers(features: nn.Module, trainable_layers: int) -> None:
    """Freeze early CNN layers and unfreeze the last N leaf modules. / Congela camadas iniciais e libera os últimos N módulos."""
    for parameter in features.parameters():
        parameter.requires_grad = False
    if trainable_layers > 0:
        leaves = [module for module in features.modules() if not list(module.children())]
        for module in leaves[-trainable_layers:]:
            for parameter in module.parameters():
                parameter.requires_grad = True


class WeightedSumAttention(nn.Module):
    """Weighted-sum temporal attention from the original notebook. / Atenção temporal por soma ponderada do notebook original."""

    def __init__(self, features: int, frames: int = 15):
        super().__init__()
        self.att_weight = nn.Parameter(torch.randn(features, 1) * 0.05)
        self.att_bias = nn.Parameter(torch.zeros(frames, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute context vector from sequence. / Calcula vetor de contexto da sequência."""
        steps = sequence.shape[1]
        bias = self.att_bias if steps == self.att_bias.shape[0] else self.att_bias[:steps]
        scores = torch.tanh(torch.matmul(sequence, self.att_weight) + bias)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights, dim=1)


class TemporalTransformerEncoder(nn.Module):
    """Temporal self-attention encoder over frame embeddings. / Codificador de autoatenção temporal sobre embeddings de quadros."""

    def __init__(self, input_dim: int, embed_dim: int, *, frames: int = 15, dropout: float = 0.25):
        super().__init__()
        self.input_size = input_dim
        self.projection = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, frames, embed_dim) * 0.02)
        nhead = next(head for head in (4, 2, 1) if embed_dim % head == 0)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence of frame vectors. / Codifica sequência temporal de vetores de quadros."""
        steps = sequence.shape[1]
        pos = self.pos_embedding[:, :steps, :] if steps <= self.pos_embedding.shape[1] else F.interpolate(
            self.pos_embedding.transpose(1, 2), size=steps, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.encoder(self.projection(sequence) + pos)


def _build_classifier(input_dim: int, dense_units: Sequence[int], dropout: float) -> nn.Sequential:
    """Build the multi-layer ReLU + Dropout classification head. / Constrói o cabeçalho de classificação ReLU + Dropout."""
    layers: list[nn.Module] = []
    in_features = input_dim
    for units in dense_units:
        layers.extend((nn.Linear(in_features, int(units)), nn.ReLU(), nn.Dropout(dropout)))
        in_features = int(units)
    layers.append(nn.Linear(in_features, 2))
    return nn.Sequential(*layers)


class ConflictModel(nn.Module):
    """Classify video sequences with CNN/ViT + BiLSTM/Transformer + optional attention. / Classifica sequências de vídeo com CNN/ViT + BiLSTM/Transformer + atenção opcional."""

    def __init__(
        self,
        *,
        backbone: str = "mobilenet_v2",
        temporal_head: str = "bilstm",
        hidden_size: int = 32,
        dropout: float = 0.25,
        pretrained: bool = True,
        trainable_layers: int = 80,
        pooling: str = "avg",
        attention: bool = True,
        dense_units: Sequence[int] = (256, 128, 64, 32),
        frames: int = 15,
        width: int = 100,
        height: int = 100,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.temporal_head_name = temporal_head
        self.features, channels = _build_backbone(backbone, pretrained)
        _configure_trainable_layers(self.features, trainable_layers)
        self.feature_dropout = nn.Dropout(dropout)
        if pooling == "avg":
            self.pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = channels
        elif pooling == "flatten":
            self.pool = nn.Identity()
            with torch.no_grad():
                sample_h = max(height, 75) if backbone == "inception_v3" else height
                sample_w = max(width, 75) if backbone == "inception_v3" else width
                feature_dim = int(self.features(torch.zeros((1, 3, sample_h, sample_w))).numel())
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}.")
        if temporal_head == "bilstm":
            self.recurrent: nn.Module = nn.LSTM(feature_dim, hidden_size, batch_first=True, bidirectional=True)
        elif temporal_head == "transformer":
            self.recurrent = TemporalTransformerEncoder(feature_dim, hidden_size * 2, frames=frames, dropout=dropout)
        else:
            raise ValueError(f"Unsupported temporal head: {temporal_head}.")
        self.attention = WeightedSumAttention(hidden_size * 2, frames=frames) if attention else None
        self.classifier = _build_classifier(hidden_size * 2, dense_units, dropout)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Return two class logits. / Retorna logits das duas classes."""
        batch, steps, channels, height, width = videos.shape
        frames = videos.reshape(batch * steps, channels, height, width)
        if self.backbone_name == "inception_v3" and (height < 75 or width < 75):
            frames = F.interpolate(frames, size=(max(height, 75), max(width, 75)), mode="bilinear", align_corners=False)
        features = self.feature_dropout(self.features(frames))
        pooled = self.pool(features).flatten(1).reshape(batch, steps, -1)
        if self.temporal_head_name == "bilstm":
            sequence, (hidden, _) = self.recurrent(pooled)
            context = self.attention(sequence) if self.attention is not None else torch.cat((hidden[0], hidden[1]), dim=1)
        else:
            sequence = self.recurrent(pooled)
            context = self.attention(sequence) if self.attention is not None else sequence.mean(dim=1)
        return self.classifier(context)
