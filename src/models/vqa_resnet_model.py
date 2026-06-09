import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetImageEncoder(nn.Module):
    def __init__(self, output_dim=256, freeze_backbone=True, train_last_block=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if train_last_block:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, image):
        features = self.backbone(image)
        features = self.projection(features)
        return features


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=256,
        padding_idx=0,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, question):
        embedded = self.embedding(question)
        _, hidden = self.gru(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        question_features = torch.cat((forward_hidden, backward_hidden), dim=-1)
        question_features = self.projection(question_features)
        return question_features


class VQAResNetModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_answers,
        image_feature_dim=256,
        question_embed_dim=256,
        question_hidden_dim=256,
        fusion_hidden_dim=512,
        dropout=0.35,
        freeze_backbone=True,
        train_last_block=True,
    ):
        super().__init__()
        self.image_encoder = ResNetImageEncoder(
            output_dim=image_feature_dim,
            freeze_backbone=freeze_backbone,
            train_last_block=train_last_block,
        )
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_dim=question_embed_dim,
            hidden_dim=question_hidden_dim,
            padding_idx=0,
            dropout=dropout,
        )

        fusion_dim = image_feature_dim + question_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim // 2, num_answers),
        )

    def forward(self, image, question):
        image_features = self.image_encoder(image)
        question_features = self.question_encoder(question)

        fused_features = torch.cat(
            [image_features, question_features],
            dim=1,
        )

        logits = self.classifier(fused_features)
        return logits
