import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#def calculate_iou(pred, target):
#    """计算IoU（Intersection over Union）"""
 #   intersection = (pred & target).float().sum()
  #  union = (pred | target).float().sum()
   # iou = (intersection + 1e-6) / (union + 1e-6)
    #return iou.item()
def calculate_iou(pred, target):
    """计算IoU（Intersection over Union）"""
    # 将预测转换为布尔类型（0或1）
    pred_bool = (pred > 0.5).bool()
    target_bool = (target > 0.5).bool()

    intersection = (pred_bool & target_bool).float().sum()
    union = (pred_bool | target_bool).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_dice(pred, target):
    """计算Dice系数"""
    pred_bool = (pred > 0.5).bool()
    target_bool = (target > 0.5).bool()

    intersection = (pred_bool & target_bool).float().sum()
    dice = (2. * intersection + 1e-6) / (pred_bool.float().sum() + target_bool.float().sum() + 1e-6)
    return dice.item()

# def calculate_dice(pred, target):
    """计算Dice系数"""
    intersection = (pred * target).float().sum()
    dice = (2. * intersection + 1e-6) / (pred.float().sum() + target.float().sum() + 1e-6)
    return dice.item()


# def calculate_metrics(preds, targets):
    """计算所有指标"""
    ious = []
    dices = []

    for pred, target in zip(preds, targets):
        ious.append(calculate_iou(pred, target))
        dices.append(calculate_dice(pred, target))

    return np.mean(ious), np.mean(dices)


def calculate_metrics(preds, targets):
    """计算所有指标"""
    ious = []
    dices = []

    # 确保preds和targets是相同形状的张量列表
    for pred, target in zip(preds, targets):
        # 如果有多批次，遍历每个样本
        if pred.dim() == 4:  # [batch, channel, height, width]
            for i in range(pred.shape[0]):
                ious.append(calculate_iou(pred[i], target[i]))
                dices.append(calculate_dice(pred[i], target[i]))
        else:  # 单个样本
            ious.append(calculate_iou(pred, target))
            dices.append(calculate_dice(pred, target))

    return np.mean(ious), np.mean(dices)

def plot_results(images, masks, preds, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))

    if len(images) == 1:
        axes = [axes]

    for i, (image, mask, pred) in enumerate(zip(images, masks, preds)):
        # 原始图像
        axes[i][0].imshow(image.transpose(1, 2, 0))
        axes[i][0].set_title('Original Image')
        axes[i][0].axis('off')

        # 真实掩码
        axes[i][1].imshow(mask.squeeze(), cmap='gray')
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')

        # 预测掩码
        axes[i][2].imshow(pred.squeeze(), cmap='gray')
        axes[i][2].set_title('Prediction')
        axes[i][2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def generate_metrics_table(preds, targets, image_names, save_path):
    """生成指标表格"""
    import pandas as pd

    results = []
    for i, (pred, target, name) in enumerate(zip(preds, targets, image_names)):
        iou = calculate_iou(pred, target)
        dice = calculate_dice(pred, target)

        # 计算准确率、召回率、F1分数
        tn, fp, fn, tp = confusion_matrix(
            target.flatten().numpy(),
            pred.flatten().numpy(),
            labels=[0, 1]
        ).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'Image': name,
            'IoU': iou,
            'Dice': dice,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    return df