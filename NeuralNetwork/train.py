import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from rich.progress import track

from my_dataset import CustomImageDataset
from convnextv2 import convnextv2_tiny


def train_one_epoch(model, optimizer, data_loader, device, loss_function, epoch):
    model.train()
    for images, labels in track(data_loader, description=f"Training Epoch {epoch}"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, data_loader, device, loss_function, epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in track(data_loader, description=f"Validating Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 实例化训练数据集
    train_dataset = CustomImageDataset(
        csv_file=args.train_data,
        img_dir=args.img_root,
        train=True,
    )

    # 实例化验证数据集
    val_dataset = CustomImageDataset(
        csv_file=args.val_data,
        img_dir=args.img_root,
        train=False,
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    model = convnextv2_tiny(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss().to(device)  # 在这里实例化损失函数

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, loss_function, epoch)
        val_loss = evaluate(model, val_loader, device, loss_function, epoch)
        print(f"Epoch {epoch}, Val Loss: {val_loss}")

        # 保存验证损失最低的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print("Saved Best Model")

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default=r'./dataset/train_dataset.csv')
    parser.add_argument('--val-data', type=str, default=r'./dataset/val_dataset.csv')
    parser.add_argument('--img-root', type=str, default=r'./dataset/images')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    # parser.add_argument('--wd', type=float, default=5e-2)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
