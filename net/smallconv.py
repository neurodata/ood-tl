import torch
import torch.nn as nn

# define the base CNN
class SmallConvSingleHeadNet(nn.Module):
    """
    Small convolution network with no residual connections
    """
    def __init__(self, num_cls=10, channels=3, avg_pool=2, lin_size=320):
        super(SmallConvSingleHeadNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size
        self.fc = nn.Linear(self.linsize, num_cls)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.flatten(1, -1)

        x = self.fc(x)
        return x
