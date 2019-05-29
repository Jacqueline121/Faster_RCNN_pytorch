import torch.nn as nn


def conv3x3(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                         nn.ReLU(inplace=True))


def maxpool2():
    return nn.MaxPool2d(stride=2)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(conv3x3(3, 64),
                                   conv3x3(3, 64),
                                   maxpool2())

        self.conv2 = nn.Sequential(conv3x3(64, 128),
                                   conv3x3(128, 128),
                                   maxpool2())

        self.conv3 = nn.Sequential(conv3x3(128, 256),
                                   conv3x3(256, 256),
                                   conv3x3(256, 256),
                                   maxpool2())

        self.conv4 = nn.Sequential(conv3x3(256, 512),
                                   conv3x3(512, 512),
                                   conv3x3(512, 512),
                                   maxpool2())

        self.conv5 = nn.Sequential(conv3x3(512, 512),
                                   conv3x3(512, 512),
                                   conv3x3(512, 512))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

