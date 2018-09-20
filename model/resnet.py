import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import pdb



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)
    
class Channel_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return y

class Spatial_Attention_Layer(nn.Module):
    def __init__(self, channel, size, reduction=8):
        super(Spatial_Attention_Layer, self).__init__()
        self.shrink = nn.Conv2d(channel, 1, 1)
        size = size**2
        self.body = nn.Sequential(
            nn.Linear(size, size//reduction),
            nn.ReLU(True),
            nn.Linear(size//reduction, size),
        )

    def forward(self, x):
        #pdb.set_trace()
        b, c, size, size = x.size()
        y = self.shrink(x).view(b, -1)
        y = self.body(y).view(b, 1, size, size)
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention_type='no_attention', size=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        if attention_type == 'channel_attention':
            self.attention = Channel_Attention_Layer(planes, reduction=4)
        elif attention_type == 'spatial_attention':
            self.attention = Spatial_Attention_Layer(planes, size, reduction=8)
        elif attention_type == 'joint_attention':
            self.attention = Joint_Attention_Layer(planes)
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
    def __init__(self, block, layers, attention_type='no_attention', num_classes=100):
        self.inplanes = 64
        self.attention_type = attention_type
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, attention_type='no_attention', size=32)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, attention_type='no_attention', size=32)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention_type=self.attention_type, size=16)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention_type=self.attention_type, size=8)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention_type='no_attention', size=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_type=attention_type, size=size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_type=attention_type, size=size))
        # append attention layer to the stage
        #layers.append(block(self.inplanes, planes, attention_type=attention_type, size=size))

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

class Fishnet(nn.Module):
    def __init__(self, block, layers, attention_type='no_attention', num_classes=100):
        self.inplanes = 64
        self.attention_type = attention_type
        super(Fishnet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, attention_type='no_attention', size=32)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, attention_type='no_attention', size=32)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention_type=self.attention_type, size=16)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention_type=self.attention_type, size=8)

        self.top_down1 = nn.Conv2d(512, 256, 1)
        self.top_down2 = nn.Conv2d(256, 256, 3, padding=1)
        self.top_down3 = nn.Conv2d(256, 256, 3, padding=1)

        self.top_down_lat1 = nn.Conv2d(256, 256, 1)
        self.top_down_lat2 = nn.Conv2d(128, 256, 1)

        self.bottom_up1 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bottom_up2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention_type='no_attention', size=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention_type=attention_type, size=size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention_type=attention_type, size=size))
        # append attention layer to the stage
        #layers.append(block(self.inplanes, planes, attention_type=attention_type, size=size))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):

        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # top down
        p5 = self.top_down1(c5)
        p4 = self._upsample_add(p5, self.top_down_lat1(c4))
        p4 = self.top_down2(p4)
        p3 = self._upsample_add(p4, self.top_down_lat2(c3))
        p3 = self.top_down3(p3)

        # bottom up
        b4 = p4 + self.bottom_up1(p3)
        b5 = c5 + self.bottom_up2(p4)

        x = self.avgpool(b5)
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

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], 'no_attention', **kwargs)
    return model

def ca_resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], 'channel_attention', **kwargs)
    return model

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def fishnet18(**kwargs):
    model = Fishnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


