"""CNN + BiLSTM + attention model aligned with the original code. / Modelo CNN + BiLSTM + atenção alinhado ao código original."""

from collections.abc import Sequence
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    DenseNet121_Weights,
    Inception_V3_Weights,
    MobileNet_V2_Weights,
    densenet121,
    inception_v3,
    mobilenet_v2,
)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Instantiate the selected CNN feature extractor. / Instancia o extrator CNN selecionado."""
    if name == "mobilenet_v2":
        net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        return net.features, int(net.last_channel)
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
    """Classify video sequences with CNN + BiLSTM + optional attention. / Classifica sequências de vídeo com CNN + BiLSTM + atenção opcional."""

    def __init__(
        self,
        *,
        backbone: str = "mobilenet_v2",
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
        self.recurrent = nn.LSTM(feature_dim, hidden_size, batch_first=True, bidirectional=True)
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
        sequence, (hidden, _) = self.recurrent(pooled)
        context = self.attention(sequence) if self.attention is not None else torch.cat((hidden[0], hidden[1]), dim=1)
        return self.classifier(context)
