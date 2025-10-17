import cv2
import numpy as np
import os
from skimage import filters, morphology, segmentation
import matplotlib.pyplot as plt


def traditional_segmentation(image_path):
    """使用传统图像处理方法分割指甲"""
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 提取饱和度通道
    saturation = hsv[:, :, 1]

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(saturation, (5, 5), 0)

    # 使用Otsu阈值法
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作（闭运算）
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建掩码
    mask = np.zeros_like(saturation)

    # 筛选最大的轮廓（假设指甲是最大的区域）
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    return mask, image


def process_directory(image_dir, output_dir):
    """处理整个目录的图像"""
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    results = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask, image = traditional_segmentation(image_path)

        # 保存结果
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, mask)

        # 可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Segmentation Result')
        axes[1].axis('off')

        plt.savefig(os.path.join(output_dir, f'vis_{image_file}'))
        plt.close()

        results.append({
            'image': image_file,
            'mask': mask
        })

    return results


if __name__ == '__main__':
    image_dir = '../data/images'
    output_dir = '../results/traditional_method'
    process_directory(image_dir, output_dir)