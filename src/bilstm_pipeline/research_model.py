"""Proposed MobileNetV2 BiLSTM model. / Modelo MobileNetV2 BiLSTM proposto."""

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ConflictModel(nn.Module):
    """Classify video sequences. / Classifica sequências de vídeo."""

    def __init__(self, *, hidden_size: int, dropout: float, pretrained: bool):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.recurrent = nn.LSTM(backbone.last_channel, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, 2))

    def forward(self, videos):
        """Return two class logits. / Retorna logits das duas classes."""
        batch, steps, channels, height, width = videos.shape
        frames = videos.reshape(batch * steps, channels, height, width)
        features = self.pool(self.features(frames)).flatten(1).reshape(batch, steps, -1)
        sequence, _ = self.recurrent(features)
        weights = torch.softmax(self.attention(sequence), dim=1)
        context = (sequence * weights).sum(dim=1)
        return self.classifier(context)
