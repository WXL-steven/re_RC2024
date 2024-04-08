import csv
import os

import numpy as np
from sklearn.model_selection import train_test_split


# def standardize_data(data):
#     # 计算均值和标准差
#     means = np.mean(data, axis=0)
#     stds = np.std(data, axis=0)
#
#     # 标准化数据
#     standardized_data = (data - means) / stds
#
#     return standardized_data, means, stds
#
#
# def split_dataset(csv_file, train_ratio=0.80, val_ratio=0.15, test_ratio=0.05):
#     # 读取CSV文件并提取必要的数据
#     rows = []
#     with open(csv_file, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             img_name = row['Image Name']
#             coeffs = [row[key] for key in ['Coeff0', 'Coeff1', 'Coeff2', 'Coeff3', 'Coeff4', 'Coeff5']]
#             rows.append([img_name] + coeffs)
#
#     # 转换为NumPy数组以便于处理
#     data = np.array(rows)
#     img_names = data[:, 0]  # 图片名
#     coeff_data = data[:, 1:].astype(float)  # Coeffs数据，转换为float类型
#
#     # 标准化Coeffs数据
#     standardized_data, means, stds = standardize_data(coeff_data)
#
#     # 更新rows中的数据
#     standardized_rows = [[img_names[i]] + list(standardized_data[i, :]) for i in range(standardized_data.shape[0])]
#
#     # 分割数据集
#     train_val, test = train_test_split(standardized_rows, test_size=test_ratio, random_state=42)
#     train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)
#
#     # 保存到新的CSV文件
#     def save_to_csv(data, filename):
#         with open(filename, 'w', newline='', encoding='utf-8') as output_file:
#             writer = csv.writer(output_file)
#             for row in data:
#                 writer.writerow(row)
#
#     save_to_csv(train, 'train_dataset.csv')
#     save_to_csv(val, 'val_dataset.csv')
#     save_to_csv(test, 'test_dataset.csv')
#
#     # 保存均值和标准差
#     with open('means_stds.csv', 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Parameter', 'Mean', 'Std'])
#         for i, key in enumerate(['Coeff0', 'Coeff1', 'Coeff2', 'Coeff3', 'Coeff4', 'Coeff5']):
#             writer.writerow([key, f"{means[i]}", f"{stds[i]}"])
#
#     print(f"Train set size: {len(train)}")
#     print(f"Validation set size: {len(val)}")
#     print(f"Test set size: {len(test)}")

def standardize_data(data):
    # 计算均值和标准差
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # 标准化数据
    standardized_data = (data - means) / stds

    return standardized_data, means, stds


def split_dataset(csv_file, train_ratio=0.80, val_ratio=0.15, test_ratio=0.05):
    # 读取CSV文件并提取必要的数据
    rows = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_name = row['Image Name']
            vector_x = row['Vector_x']
            rows.append([img_name, vector_x])

    # 转换为NumPy数组以便于处理
    data = np.array(rows)
    img_names = data[:, 0]  # 图片名
    vector_data = data[:, 1].astype(float)  # Vector_x数据，转换为float类型

    # 标准化Vector_x数据
    standardized_data, mean, std = standardize_data(vector_data.reshape(-1, 1))

    # 更新rows中的数据
    standardized_rows = [[img_names[i]] + list(standardized_data[i, :]) for i in range(standardized_data.shape[0])]

    # 分割数据集
    train_val, test = train_test_split(standardized_rows, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    # 保存到新的CSV文件
    def save_to_csv(data, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            for row in data:
                writer.writerow(row)

    save_to_csv(train, 'train_dataset.csv')
    save_to_csv(val, 'val_dataset.csv')
    save_to_csv(test, 'test_dataset.csv')

    # 保存均值和标准差
    with open('means_stds.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Mean', 'Std'])
        writer.writerow(['Vector_x', f"{mean[0]}", f"{std[0]}"])

    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")


if __name__ == '__main__':
    # 调用函数
    csv_file = r'C:\Users\Steven\Downloads\QQ\data.csv'  # CSV文件的路径
    # img_dir = r"C:\Users\Steven\Downloads\QQ\pic"  # 图像目录的路径
    split_dataset(csv_file)
