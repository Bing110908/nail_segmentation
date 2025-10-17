import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import NailSegmentationModel
import os
import argparse


class NailSegmenter:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.image_size = 256

    def load_model(self, model_path):
        """加载训练好的模型"""
        model = NailSegmentationModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        """预处理图像"""
        if len(image.shape) == 2:  # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        original_shape = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)

        return image_tensor, image_rgb, original_shape

    def segment(self, image_path, threshold=0.5):
        """分割指甲区域"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        image_tensor, original_image, original_shape = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = (output > threshold).float()

        # 后处理
        prediction_np = prediction.squeeze().cpu().numpy()
        prediction_resized = cv2.resize(prediction_np, (original_shape[1], original_shape[0]))

        return original_image, prediction_resized

    def create_overlay(self, image, mask, alpha=0.5):
        """创建叠加可视化"""
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [255, 0, 0]  # 红色显示指甲区域

        # 叠加
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlay


def main():
    parser = argparse.ArgumentParser(description='指甲分割预测')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='./results/cnn_model/best_model.pth',
                        help='模型路径')
    parser.add_argument('--output_dir', type=str, default='./results/predictions',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分割器
    segmenter = NailSegmenter(args.model_path)

    # 进行分割
    try:
        original_image, mask = segmenter.segment(args.image_path)

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]

        # 保存结果
        output_paths = {
            'original': os.path.join(args.output_dir, f'{base_name}_original.png'),
            'mask': os.path.join(args.output_dir, f'{base_name}_mask.png'),
            'overlay': os.path.join(args.output_dir, f'{base_name}_overlay.png'),
            'comparison': os.path.join(args.output_dir, f'{base_name}_comparison.png')
        }

        # 保存原始图像
        cv2.imwrite(output_paths['original'], cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

        # 保存掩码
        cv2.imwrite(output_paths['mask'], (mask * 255).astype(np.uint8))

        # 保存叠加结果
        overlay = segmenter.create_overlay(original_image, mask)
        cv2.imwrite(output_paths['overlay'], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Nail Mask')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Result')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_paths['comparison'], bbox_inches='tight', dpi=300)
        plt.close()

        print("分割完成！")
        for name, path in output_paths.items():
            print(f"{name}: {path}")

    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == '__main__':
    main()