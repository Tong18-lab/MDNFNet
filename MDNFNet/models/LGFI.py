import torch
import torch.nn as nn
import torch.nn.functional as F

class LGFI_Module(nn.Module):
    def __init__(self, in_channels, patch_size=6):
        super(LGFI_Module, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 提取全局身份特征
        self.local_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        global_feat = self.global_pool(x)                  # (B, C, 1, 1)
        global_feat_exp = global_feat.expand_as(x)         # (B, C, H, W)

        local_feat = self.local_conv(x)                    # (B, C, H, W)

        concat = torch.cat([local_feat, global_feat_exp], dim=1)  # (B, 2C, H, W)
        attn = self.attn_mlp(concat)                       # (B, C, H, W)

        out = x * attn                                     # 局部增强，全局引导
        return out