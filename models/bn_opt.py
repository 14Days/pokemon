import torch.nn as nn
import torch.optim as optim

config_feature = [32, 'M', 'B', 64, 'M', 'B', 64, 'M', 'B', 128, 'M']
config_classifier = ['F', 'B', 'D', 'F', 'B', 'F']


class BnOpt(nn.Module):
    def __init__(self, output):
        super(BnOpt, self).__init__()
        self._make_feature_layers()
        self._make_classifier_layers(output)

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_feature_layers(self):
        layers = []
        in_channels = 3
        for index, item in enumerate(config_feature):
            if item == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif item == 'B':
                layers += [nn.BatchNorm2d(config_feature[index - 2])]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels, item, kernel_size=3,
                    ),
                    nn.ReLU(inplace=True)
                ]
                in_channels = item

        self.feature = nn.Sequential(*layers)

    def _make_classifier_layers(self, output):
        layers = [
            nn.Linear(12800, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output),
            nn.LogSoftmax(dim=1)
        ]
        self.classifier = nn.Sequential(*layers)


class MyBnOpt:
    def __init__(self, output, lr):
        self.model = BnOpt(output)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, dampening=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)


if __name__ == '__main__':
    net = MyBnOpt(2, 0.001)
    print(net.model)
