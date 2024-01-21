import torch.nn as nn


class ResidualBlock(nn.Module):
    pass


class CNNv1(nn.Module):
    def __init__(self, classes):
        super(CNNv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=13, stride=2, dilation=3, padding=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2,
            downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128)))
        self.layer3 = ResidualBlock(128, 256, stride=2,
            downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2), nn.BatchNorm2d(256)))
        self.layer4 = ResidualBlock(256, 512, stride=2,
            downsample=nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2), nn.BatchNorm2d(512)))
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=4, stride=1, dilation=1, padding=2)
        self.bn2 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
