import torch.nn as nn
import torch.optim as optim
from torchvision import models


class MyVGG:
    def __init__(self, output):
        # 得到神经网络
        self.model = models.vgg16(pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output),
            nn.LogSoftmax(dim=1)
        )

        # 得到loss、优化器等
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(0.001), lr=0.0001, momentum=0.9, dampening=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)


if __name__ == '__main__':
    vgg = MyVGG(10)
