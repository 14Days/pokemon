import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data import get_data_loader
from config import get_save_path


class MyVGG:
    def __init__(self, kind, mode, batch_size, epoch):
        self.start_epoch = 0
        self.epoch = epoch
        self.kind = kind
        # 得到数据集
        self.loader, self.dataset = get_data_loader(kind, mode, batch_size)
        # 得到神经网络
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(self.dataset.name2label.keys())),
            nn.LogSoftmax(dim=1)
        )

        # 得到loss、优化器等
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(0.001), lr=0.001, momentum=0.9, dampening=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        # 加载参数
        if os.path.exists(get_save_path(kind)):
            checkpoint = torch.load(dir)
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1

    def get_params_number(self):
        """
        输出模型的参数
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} 参数总数.')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} 可训练参数总数.')

    def train(self):
        """
        训练模型
        """
        for index in range(self.start_epoch, self.epoch):
            loss_sigma = 0.0  # 记录一个 epoch 的 loss 之和
            correct = 0.0
            total = 0.0

            for i, data in enumerate(self.loader):
                # 获取图片和标签
                inputs, labels = data

                # forward, backward, update weights
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()

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
                        self.epoch + 1, self.epoch, i + 1, len(self.loader), loss_avg, correct / total))
                    torch.save({
                        'net': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': index + 1
                    }, get_save_path(self.kind))

            # 更新学习率
            self.scheduler.step(self.epoch)


if __name__ == '__main__':
    vgg = MyVGG('color', 'train', 32, 10)
    vgg.get_params_number()
    vgg.train()
