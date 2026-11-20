# ========================================================================================
# IndicDLP: Neural Network Architecture Definitions
# 1. LeNet-5 (Modified for 224x224 RGB inputs)
# 2. ResNet-18 (Deep Residual Backbone with Global Average Pooling)
# ========================================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedLeNet5(nn.Module):
    """
    Classical LeNet-5 adapted for 224x224 RGB document image inputs.
    Uses 3 Conv-Pool blocks, Adaptive Average Pooling, and a 2-layer classifier.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112

            # Block 2
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56

            # Block 3
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28
        )
        
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_resnet18(num_classes=2, pretrained=True):
    """
    Builds ResNet-18 backbone with ImageNet pretrained weights and binary classification head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
