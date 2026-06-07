import torch 
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
        )

        self.projection = nn.Linear(256, output_dim)

    def forward(self, image):
        features = self.cnn(image)
        features = features.flatten(1)
        features = self.projection(features)
        return features

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True
        )
    def forward(self, question):
        embedded = self.embedding(question)
        output, _ = self.gru(embedded)
        return output

class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_answers,
        image_feature_dim=256,
        question_embed_dim=128,
        question_hidden_dim=256,
        fusion_hidden_dim=512,
        dropout=0.3,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            output_dim=image_feature_dim,
        )
        self.question_encoder = QuestionEncoder(
            vocab_size, question_embed_dim, question_hidden_dim, padding_idx=0
        )
        fusion_dim = image_feature_dim + question_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_answers),
        )

    def forward(self, image, question):
        image_features = self.image_encoder(image)
        question_features = self.question_encoder(question)
        fused_features = torch.cat([image_features, question_features], dim=1)
        logits = self.classifier(fused_features)
        return logits
