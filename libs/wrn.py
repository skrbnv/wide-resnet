import torch.nn as nn


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, stride=stride)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, stride=1)
        self.shortcut = nn.Identity() if in_planes == planes else nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        c = self.bn1(x)
        c = self.relu(c)
        c = self.conv1(c)

        c = self.bn2(c)
        c = self.relu(c)
        if self.dropout is not None:
            c = self.dropout(c)
        c = self.conv2(c)
        c += self.shortcut(x)
        return c


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = conv3x3(in_planes=3, out_planes=nStages[0], stride=1)
        self.layer1 = self.create_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self.create_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self.create_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        self.pool = nn.AvgPool2d(8)
        self.relu = nn.ReLU()
        self.__init_wb__()

    def __init_wb__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def create_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        extracted = out.clone()
        out = self.pool(out)
        out = self.linear(out.flatten(1))
        return out, extracted
