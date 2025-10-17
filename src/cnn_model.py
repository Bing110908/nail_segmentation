import torch
import torch.nn as nn
from torchvision import models
from segmentation_models_pytorch import Unet


class NailSegmentationModel(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
        super(NailSegmentationModel, self).__init__()

        self.model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)


# 自定义损失函数（Dice Loss + BCE Loss）
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # Dice系数计算
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss