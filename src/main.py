import argparse
import torch
from train import main as train_main
from evaluate import calculate_metrics, plot_results, generate_metrics_table
from traditional_method import process_directory as traditional_process
from dataset import NailDataset
from cnn_model import NailSegmentationModel
import os


def main():
    parser = argparse.ArgumentParser(description='Nail Segmentation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'traditional'],
                        help='Mode: train, evaluate, or traditional')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing images and masks')
    parser.add_argument('--model_path', type=str, default='./results/cnn_model/best_model.pth',
                        help='Path to trained model for evaluation')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')

    args = parser.parse_args()

    if args.mode == 'train':
        train_main()

    elif args.mode == 'evaluate':
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NailSegmentationModel()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        # 加载数据
        image_dir = os.path.join(args.data_dir, 'images')
        mask_dir = os.path.join(args.data_dir, 'masks')
        dataset = NailDataset(image_dir, mask_dir, transform=None, is_train=False)

        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # 评估模型
        all_preds = []
        all_targets = []
        all_images = []

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = (outputs > 0.5).float()

                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                all_images.append(images.cpu())

        # 计算指标
        iou, dice = calculate_metrics(all_preds, all_targets)
        print(f'IoU: {iou:.4f}, Dice: {dice:.4f}')

        # 可视化一些结果
        os.makedirs(os.path.join(args.output_dir, 'evaluation'), exist_ok=True)
        for i in range(min(5, len(all_images))):
            plot_results(
                [all_images[i].squeeze().numpy()],
                [all_targets[i].squeeze().numpy()],
                [all_preds[i].squeeze().numpy()],
                save_path=os.path.join(args.output_dir, 'evaluation', f'result_{i}.png')
            )

        # 生成指标表格
        image_names = [f'image_{i}' for i in range(len(all_images))]
        metrics_table = generate_metrics_table(
            all_preds, all_targets, image_names,
            save_path=os.path.join(args.output_dir, 'evaluation', 'metrics.csv')
        )

        print(metrics_table.describe())

    elif args.mode == 'traditional':
        image_dir = os.path.join(args.data_dir, 'images')
        output_dir = os.path.join(args.output_dir, 'traditional_method')
        traditional_process(image_dir, output_dir)


if __name__ == '__main__':
    main()