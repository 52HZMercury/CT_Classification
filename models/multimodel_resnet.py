import torch
import torch.nn as nn
from monai.networks.nets import ResNet, resnet18


class MultiModelResNet(nn.Module):
    def __init__(self, tabular_dim, in_channels, num_classes=2):
        super(MultiModelResNet, self).__init__()

        # 1. 影像分支：使用 ResNet18 提取空间特征
        # 注意 in_channels=3
        self.resnet = resnet18(
            pretrained=False,
            n_input_channels=in_channels,
            num_classes=512  # 让它输出一个 512 维的特征向量
        )
        # 移除 resnet 最后的 fc 层，我们自己做融合
        self.resnet.fc = nn.Identity()

        # 2. 临床信息分支：简单的 MLP
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 3. 融合后的分类器
        # 512 (图像特征) + 64 (临床特征) = 576
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, tab):
        # 提取影像特征: [B, 512]
        img_feat = self.resnet(img)

        # 提取临床特征: [B, 64]
        tab_feat = self.tabular_mlp(tab)

        # 拼接特征: [B, 576]
        combined_feat = torch.cat((img_feat, tab_feat), dim=1)

        # 输出分类结果
        logits = self.classifier(combined_feat)
        return logits