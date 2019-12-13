import torch.nn as nn
import torch.optim as optim

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, output):
        super(VGG, self).__init__()
        self._make_feature_layers(cfg[vgg_name])
        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self._make_classifier_layers(output)

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def _make_feature_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels, x, kernel_size=3,
                        padding=1, bias=False
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        self.features = nn.Sequential(*layers)

    def _make_classifier_layers(self, output):
        layers = [
            nn.Linear(25088, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output),
            nn.LogSoftmax(dim=1)
        ]
        self.classifier = nn.Sequential(*layers)


class MyVGG:
    def __init__(self, name, output, lr):
        # 得到神经网络
        self.model = VGG(name, output)

        # 得到loss、优化器等
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, dampening=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)


if __name__ == '__main__':
    vgg = MyVGG('VGG16', 10, 0.0001)
    print(vgg.model, vgg.optimizer)
