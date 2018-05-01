import torch.nn as nn
import math
import torch


def get_model(name):
    if name == 'resnet':
        return resnet50()
    elif name == 'ca_resnet':
        return ca_resnet50()
    elif name == 'sa_resnet':
        return sa_resnet50()
    elif name == 'ja_resnet':
        return ja_resnet50()
    else:
        raise Exception('Unknown model ', name)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)
    
class Channel_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

#class Spatial_Attention_Layer(nn.Module):
#    def __init__(self, channel):
#        super(Spatial_Attention_Layer, self).__init__()
#        self.conv1 = nn.Conv2d(channel, 1, 1)
#        self.relu = nn.ReLU(True)
#        self.localization = nn.Sequential(
#            nn.Conv2d(1, 8, kernel_size=7),
#            nn.MaxPool2d(2, stride=2),
#            nn.ReLU(True),
#            nn.Conv2d(8, 10, kernel_size=5),
#            nn.MaxPool2d(2, stride=2),
#            nn.ReLU(True)
#        )
#
#        # Regressor for the 3 * 2 affine matrix
#        self.fc_loc = nn.Sequential(
#            nn.Conv2d(
#            nn.Linear(10 * 4 * 4, 32),
#            nn.ReLU(True),
#            nn.Linear(32, 3 * 2)
#        )
#
#        # Initialize the weights/bias with identity transformation
#        self.fc_loc[2].weight.data.zero_()
#        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#    def forward(self, x):
#        x = self.relu(self.conv1(x))
#        xs = self.localization(x)
#        xs = xs.view(-1, 10 * 4 * 4)
#        theta = self.fc_loc(xs)
#        theta = theta.view(-1, 2, 3)
#
#        grid = F.affine_grid(theta, x.size())
#        x = F.grid_sample(x, grid)
#
#        return x

class Spatial_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Spatial_Attention_Layer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel//reduction, channel//reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel//reduction, 1, 1, bias=True)
        )

    def forward(self, x):
        y = self.body(x)
        return y

class Joint_Attention_Layer(nn.Module):
    def __init__(self, channel):
        super(Joint_Attention_Layer, self).__init__()
        self.channel_attention = Channel_Attention_Layer(channel)
        self.spatial_attention = Spatial_Attention_Layer(channel)

    def forward(self, x):
        y1 = self.channel_attention(x)
        y2 = self.spatial_attention(x)
        return y1 + y2

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='no_attention'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        if attention_type == 'channel_attention':
            self.attention = Channel_Attention_Layer(planes*4)
        elif attention_type == 'spatial_attention':
            self.attention = Spatial_Attention_Layer(planes*4)
        elif attention_type == 'joint_attention':
            self.attention = Joint_Attention_Layer(planes*4)
        elif attention_type == 'no_attention':
            self.attention = None
        else:
            raise Exception('Unknown attention type')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attention is not None:
            attention = self.attention(out)
            attention = self.sigmoid(attention)
            out = out*attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, attention_type, num_classes=100):
        self.inplanes = 64
        self.attention_type = attention_type
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, attention_type=self.attention_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, attention_type=self.attention_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention_type=self.attention_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention_type='no_attention'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        # append attention layer to the stage
        layers.append(block(self.inplanes, planes, attention_type=attention_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ca_resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'channel_attention', **kwargs)
    return model


def sa_resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'spatial_attention', **kwargs)
    return model

def ja_resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'joint_attention', **kwargs)
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], 'no_attention', **kwargs)
    return model


