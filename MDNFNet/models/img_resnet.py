import torchvision
import torch
from torch import nn
from torch.nn import init
from models import pooling


import torchvision.models as torchvision_models
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))
class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class LiteMSA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1, padding=0)
        self.branch2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.branch3 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=5, padding=2)

        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.eca = ECAAttention(in_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.fusion(x)
        x = self.eca(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)

        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)

        self.layer0 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        self.layer1 = resnet50.layer1  # Output: 256
        self.layer2 = resnet50.layer2  # Output: 512
        self.msa2 = LiteMSA(512)

        self.layer3 = resnet50.layer3  # Output: 1024
        self.msa3 = LiteMSA(1024)

        self.layer4 = resnet50.layer4  # Output: 2048

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.msa3(x)
        x = self.layer4(x)
        return x
class ResNet101(nn.Module):
    def __init__(self, config):
        super(ResNet101, self).__init__()
        resnet101 = torchvision.models.resnet101(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet101.layer4[0].conv2.stride = (1, 1)
            resnet101.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])

    def forward(self, x):
        x = self.base(x)
        return x
#
class Part_Block(nn.Module):
    def __init__(self, in_channels=2048, num_parts=4, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_parts = num_parts

        # 动态卷积权重生成
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.silu = nn.SiLU(inplace=True)  # 这里用 SiLU 替换 ReLU
        self.fc2 = nn.Linear(in_channels // reduction, in_channels * num_parts)

        # Base卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)

        # 输出
        self.final_conv = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # ------- 动态卷积权重生成 -------
        pooled = self.global_pool(x).view(B, C)  # [B, C]
        hidden = self.silu(self.fc1(pooled))     # 用 SiLU 代替 ReLU
        dynamic_weight = self.fc2(hidden)        # [B, C*num_parts]
        dynamic_weight = dynamic_weight.view(B, self.num_parts, C, 1, 1)  # [B, num_parts, C, 1, 1]

        # ------- 应用动态卷积 -------
        base_feature = self.base_conv(x)  # [B, C, H, W]

        out = []
        for i in range(self.num_parts):
            weight = dynamic_weight[:, i, :, :, :]  # [B, C, 1, 1]
            weighted = base_feature * weight
            weighted = weighted.sum(dim=1, keepdim=True)  # [B, 1, H, W]
            out.append(weighted)

        out = torch.cat(out, dim=1)  # [B, num_parts, H, W]

        # ------- 输出 soft attention maps -------
        out = self.softmax(out)
        return out


class GAP_Classifier(nn.Module):

    def __init__(self, config, num_identities):
        super().__init__()
        self.bn = nn.BatchNorm2d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(config.MODEL.FEATURE_DIM, num_identities, kernel_size=1)
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.classifier = nn.Linear(config.MODEL.FEATURE_DIM, num_identities)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):

        x = self.bn(x)
        x = self.conv(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)

        return x
