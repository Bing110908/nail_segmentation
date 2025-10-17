import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from dataset import get_dataloaders
from cnn_model import NailSegmentationModel, CombinedLoss
from evaluate import calculate_metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_iou, val_dice = evaluate_model(model, val_loader, criterion, device)

        # 更新学习率
        if scheduler:
            scheduler.step(val_loss)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)

        print(
            f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}')

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))

    writer.close()
    return model


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # 收集预测和目标用于计算指标
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    # 计算指标
    epoch_loss = running_loss / len(data_loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    iou, dice = calculate_metrics(all_preds, all_targets)

    return epoch_loss, iou, dice

def main():
    # 设置参数
    data_dir = '../data'
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    batch_size = 2
    num_epochs = 90
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(image_dir, mask_dir, batch_size)

    # 初始化模型
    model = NailSegmentationModel()
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 训练模型
    save_dir = '../results/cnn_model'
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir)


if __name__ == '__main__':
    main()