import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision import models
from data import get_data_loader


def get_vgg16(output):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, output),
        nn.LogSoftmax(dim=1)
    )
    return model


def train_model(kind, mode, batch_size, epoch):
    # 得到模型
    loader, dataset = get_data_loader(kind, mode, batch_size)
    model = get_vgg16(len(dataset.name2label.keys()))
    print(model)

    # 输出模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} 参数总数.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} 可训练参数总数.')

    # 定义优化器、loss
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(0.001), lr=0.001, momentum=0.9, dampening=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for _ in range(epoch):
        loss_sigma = 0.0  # 记录一个 epoch 的 loss 之和
        correct = 0.0
        total = 0.0

        for i, data in enumerate(loader):
            # if i == 30 : break
            # 获取图片和标签
            inputs, labels = data

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()
            loss_sigma += loss.item()

            # 每 10 个 iteration 打印一次训练信息，loss 为 10 个 iteration 的平均
            print(i)
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, epoch, i + 1, len(loader), loss_avg, correct / total))

        # 更新学习率
        scheduler.step(epoch)


if __name__ == '__main__':
    train_model('color', 'train', 32, 10)
