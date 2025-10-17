import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


class NailDataset(Dataset):
    def __init__(self, image_dir, mask_dir, specific_images=None, transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train

        if specific_images is None:
            # 使用所有图片
            self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        else:
            # 使用指定的图片列表
            self.images = specific_images

        # 智能匹配掩码文件
        self.masks = self._find_matching_masks()

        print(f"✅ 找到 {len(self.images)} 张图像和 {len(self.masks)} 个掩码")

    def _find_matching_masks(self):
        """智能查找对应的掩码文件"""
        masks = []
        valid_images = []

        # 获取掩码目录中的所有文件
        mask_files = set([f for f in os.listdir(self.mask_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        for img_file in self.images:
            # 尝试多种可能的掩码文件名格式
            possible_mask_names = [
                img_file,  # 同名文件
                img_file.replace('.jpg', '.png').replace('.jpeg', '.png'),
                img_file.replace('.png', '.jpg'),
                img_file.replace('.jpeg', '.jpg'),
                os.path.splitext(img_file)[0] + '.png',  # 相同基础名，不同扩展名
                os.path.splitext(img_file)[0] + '.jpg',
            ]

            found_mask = None
            for mask_name in possible_mask_names:
                if mask_name in mask_files:
                    found_mask = mask_name
                    break

            if found_mask:
                masks.append(found_mask)
                valid_images.append(img_file)
            else:
                print(f"⚠️ 警告: 未找到图像 {img_file} 对应的掩码文件")

        # 更新有效的图像列表
        self.images = valid_images

        if len(self.images) == 0:
            raise ValueError("没有找到有效的图像-掩码对！")

        return masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码: {mask_path}")

        # 统一图像和掩码尺寸
        target_size = (256, 256)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 转换为PyTorch张量
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)


def get_dataloaders(image_dir, mask_dir, batch_size=2, val_split=0.2):
    # 获取所有图像文件
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(f"📁 找到图像文件: {len(all_images)}个")

    # 划分训练集和验证集
    train_images, val_images = train_test_split(all_images, test_size=val_split, random_state=42)

    print(f"🎯 数据集划分:")
    print(f"  训练集: {len(train_images)}个图像")
    print(f"  验证集: {len(val_images)}个图像")

    # 创建数据集
    print("🔄 加载训练集...")
    train_dataset = NailDataset(image_dir, mask_dir, specific_images=train_images, is_train=True)

    print("🔄 加载验证集...")
    val_dataset = NailDataset(image_dir, mask_dir, specific_images=val_images, is_train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 最终数据统计:")
    print(f"  训练批次: {len(train_loader)}个 (共{len(train_dataset)}张图片)")
    print(f"  验证批次: {len(val_loader)}个 (共{len(val_dataset)}张图片)")

    return train_loader, val_loader