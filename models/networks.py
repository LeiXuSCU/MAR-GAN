from typing import Type, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.vision.resnet_vision import ResNet


def resnet_backbone(model_type: str = 'resnet101'):
    block: Type[Union[torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck]]
    layers: List[int]
    if model_type == 'resnet18':
        block = torchvision.models.resnet.BasicBlock
        layers = [2, 2, 2, 2]
    elif model_type == 'resnet34':
        block = torchvision.models.resnet.BasicBlock
        layers = [3, 4, 6, 3]
    elif model_type == 'resnet50':
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 6, 3]
    elif model_type == 'resnet101':
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 23, 3]
    elif model_type == 'resnet152':
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 8, 36, 3]
    # return ResNet(block=block, layers=layers, in_channels=in_channels)
    return block, layers


class BackboneResnet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 model_type: str = 'resnet101'):
        super(BackboneResnet, self).__init__()

        block, layers = resnet_backbone(model_type)
        self.backbone = ResNet(block=block, layers=layers, in_channels=in_channels)
        # self.backbone = resnet_backbone(in_channels, model_type)
        up_in_planes = 512 * (block.expansion + 2)
        self.up_layer4 = UpConv(up_in_planes, block, 256, layers[3])
        up_in_planes = 256 * (block.expansion + 2)
        self.up_layer3 = UpConv(up_in_planes, block, 128, layers[2])
        up_in_planes = 128 * (block.expansion + 2)
        self.up_layer2 = UpConv(up_in_planes, block, 64, layers[1])
        up_in_planes = 64 + 64 * block.expansion
        self.up_layer1 = UpConv(up_in_planes, block, 16, layers[0])
        self.out_layer = LastUpConv(64, in_channels)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input -- {}'.format(x.shape))
        layer0, layer1, layer2, layer3, layer4 = self.backbone(x)
        # print('layer0 -- {}'.format(layer0.shape))
        # print('layer1 -- {}'.format(layer1.shape))
        # print('layer2 -- {}'.format(layer2.shape))
        # print('layer3 -- {}'.format(layer3.shape))
        # print('layer4 -- {}'.format(layer4.shape))
        up_layer4 = self.up_layer4(layer4, layer3)
        # print('up_layer4 -- {}'.format(up_layer4.shape))
        up_layer3 = self.up_layer3(up_layer4, layer2)
        # print('up_layer3 -- {}'.format(up_layer3.shape))
        up_layer2 = self.up_layer2(up_layer3, layer1)
        # print('up_layer2 -- {}'.format(up_layer2.shape))
        up_layer1 = self.up_layer1(up_layer2, layer0)
        # print('up_layer1 -- {}'.format(up_layer1.shape))
        out = self.out_layer(up_layer1)
        # print('out before tanh -- {}'.format(out))
        out = torch.tanh(out)
        # print('out after tanh -- {}'.format(out))
        return out


class LastUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastUpConv, self).__init__()
        self.conv = nn.Sequential(BackwardTransition(in_channels, in_channels),
                                  BackwardTransition(in_channels, in_channels),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.conv(x)
        return x


class BackwardTransition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BackwardTransition, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=True)
        return x


class UpConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 block: Type[Union[torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck]],
                 planes: int,
                 blocks: int,
                 stride: int = 1):
        super(UpConv, self).__init__()
        self.up = make_layer(in_planes, block, planes, blocks, stride=stride)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2), dim=1)
        x = self.up(x)
        return x


def make_layer(in_planes,
               block: Type[Union[torchvision.models.resnet.BasicBlock, torchvision.models.resnet.Bottleneck]],
               planes: int,
               blocks: int,
               stride: int = 1) -> nn.Sequential:
    down_sample = None
    if stride != 1 or in_planes != planes * block.expansion:
        down_sample = nn.Sequential(
            torchvision.models.resnet.conv1x1(in_planes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = [block(in_planes, planes, stride, down_sample, 1, 64, 1, nn.BatchNorm2d)]
    in_planes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(in_planes, planes, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 ndf: int = 64):
        super(Discriminator, self).__init__()

        self.net = [
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=0, bias=False)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
