import os
import torch
import torch.nn as nn
from models.vgg import MyVGG
from data import get_data_loader
from config import get_save_path


class Net:
    def __init__(self, kind, mode, batch_size, epoch):
        self.start_epoch = 0
        self.epoch = epoch
        self.kind = kind
        self.path = get_save_path(kind)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 得到数据集
        self.loader, self.dataset = get_data_loader(kind, mode, batch_size)

        # 得到神经网络
        self.model = MyVGG(len(self.dataset.name2label.keys()))

        if torch.cuda.device_count() > 1:
            print(f'We use {torch.cuda.device_count()} GPUs')
            self.model.model = nn.DataParallel(self.model.model)

        self.model.model.to(self.device)

        # 加载参数
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path)
            self.model.model.load_state_dict(checkpoint['net'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.start_epoch = checkpoint['epoch'] + 1

    def get_params_number(self):
        """
        输出模型的参数
        """
        total_params = sum(p.numel() for p in self.model.model.parameters())
        print(f'{total_params:,} 参数总数.')
        total_trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} 可训练参数总数.')

    def train(self):
        """
        训练模型
        """
        max_correct = 0.0
        for index in range(self.start_epoch, self.epoch):
            loss_sigma = 0.0  # 记录一个 epoch 的 loss 之和
            correct = 0.0
            total = 0.0

            for i, data in enumerate(self.loader):
                # 获取图片和标签
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward, backward, update weights
                self.model.optimizer.zero_grad()
                outputs = self.model.model(inputs)
                loss = self.model.loss_func(outputs, labels)
                loss.backward()
                self.model.optimizer.step()

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
                loss_sigma += loss.item()

                # 每 10 个 iteration 打印一次训练信息，loss 为 10 个 iteration 的平均
                if i % 10 == 9:
                    loss_avg = loss_sigma / 10
                    loss_sigma = 0.0
                    print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        index, self.epoch, i + 1, len(self.loader), loss_avg, correct / total))
                    if correct > max_correct:
                        max_correct = correct
                        torch.save({
                            'net': self.model.model.state_dict(),
                            'optimizer': self.model.optimizer.state_dict(),
                            'epoch': index
                        }, self.path)

            # 更新学习率
            self.model.scheduler.step(self.epoch)

    def test(self):
        loss_sigma = 0.0
        total = 0.0
        correct = 0.0
        self.model.model.eval()
        for i, data in enumerate(self.loader):
            # 获取图片和标签
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            # forward
            outputs = self.model.model(images)
            outputs.detach_()

            # 计算loss
            loss = self.model.loss_func(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()

        print('{} set Accuracy:{:.2%}'.format('Valid', correct / total))


if __name__ == '__main__':
    vgg = Net('color', 'train', 32, 10)
    vgg.get_params_number()
    vgg.train()
