from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, train=True):
        """
        Args:
            csv_file (string): CSV文件的路径。
            img_dir (string): 图像文件夹的路径。
            train (bool): 指示是否为训练集，用于决定是否应用颜色增强。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        if train:
            # 仅在训练集上应用颜色增强
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色增强
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # 测试集（或验证集）不应用颜色增强
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1:].values.astype('float32')
        image = self.transform(image)
        return image, label
