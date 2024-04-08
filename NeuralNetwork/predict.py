import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import argparse
import os

# 假设CustomImageDataset是您定义的数据集类
from my_dataset import CustomImageDataset
from convnextv2 import convnextv2_tiny


def load_normalization_parameters(csv_file):
    """从CSV文件加载归一化参数"""
    df = pd.read_csv(csv_file)
    means = df['Mean'].values
    stds = df['Std'].values
    return means, stds


def unnormalize(predictions, means, stds):
    """还原预测数据"""
    predictions = predictions * stds + means
    return predictions


def predict_image(model, image_path, transform):
    """对单个图像进行预测"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    with torch.no_grad():
        outputs = model(image)
    return outputs.squeeze(0)  # 移除batch维度


def main():
    parser = argparse.ArgumentParser(description="Predict script")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--norm_params", type=str, required=True, help="Path to the normalization parameters CSV")
    parser.add_argument("--img_root", type=str, required=True, help="Root directory for the images")
    parser.add_argument("--test_data", type=str, required=True, help="CSV file with training data information")
    args = parser.parse_args()

    # 加载模型和权重到CPU
    model = convnextv2_tiny(num_classes=6)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    # 加载归一化参数
    means, stds = load_normalization_parameters(args.norm_params)

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    dataset = CustomImageDataset(
        csv_file=args.test_data,
        img_dir=args.img_root,
        train=False
    )

    for idx in range(len(dataset)):
        image_path, expected = dataset.img_labels.iloc[idx, 0], dataset.img_labels.iloc[idx, 1:].values
        image_path = os.path.join(args.img_root, image_path)

        # 预测
        predictions = predict_image(model, image_path, transform)
        predictions = predictions.numpy()  # 转换为numpy数组
        predictions = unnormalize(predictions, means, stds)

        # 还原图像并展示
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 480))
        # 计算
        poly = np.poly1d(predictions)
        y_vals = np.linspace(0, image.shape[0] - 1, 100)
        x_vals = np.polyval(poly, y_vals).astype(int)
        curve_points = np.array([x_vals, y_vals]).T.reshape((-1, 1, 2))
        curve_points = curve_points.astype(np.int32)
        cv2.polylines(image, [curve_points], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow("Image", image)
        try:
            cv2.waitKey(0)  # 等待按键
        except KeyboardInterrupt:
            break

        # 打印预测结果和期望结果
        print(f"Predicted: {predictions}, Expected: {unnormalize(expected, means, stds)}")


if __name__ == "__main__":
    main()
