import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist
from monai.transforms import AsDiscrete
from monai.networks.utils import one_hot


# ================== 核心新增代码：Focal Loss 类 ==================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        分类专用的 Focal Loss
        :param alpha: 类别权重列表，例如二分类不平衡可以设为 [0.2, 0.8]
        :param gamma: 难易样本调节参数，默认 2.0。值越大，模型越关注困难样本
        :param reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            # 将传入的 alpha 列表转换为 Tensor
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 形状 [Batch, Num_Classes] (Logits)
        # targets: 形状 [Batch] (类别的整数索引)

        # 将 alpha 移动到与输入相同的设备上 (GPU/CPU)
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None

        # 计算基础的交叉熵损失 (不求平均，保留每个样本的 loss)
        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')

        # 计算预测概率 pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        # 应用 Focal Loss 公式: (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =================================================================


class Trainer:
    def __init__(self, config):
        """
        初始化训练器
        """
        self.config = config

        self.device = self._init_device()
        self.train_loader, self.val_loaders = self._init_data_loaders()
        self.val_loader = next(iter(self.val_loaders.values()))
        self.model = self._init_model().to(self.device)

        # 分类专用损失函数
        self.loss_function = self._init_loss()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        log_dir = os.path.join(self.config['data']['out_dir'], self.config['data']['exp_name'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.config['model']['num_classes'])
        self.post_label = AsDiscrete(to_onehot=self.config['model']['num_classes'])

        self.metric = ConfusionMatrixMetric(
            metric_name=["accuracy", "precision", "recall", "specificity", "negative_predictive_value"],
            compute_sample=False,
            include_background=True
        )
        self.auc_metric = ROCAUCMetric()

        self.global_step = 0
        self.epoch_loss_values = []
        self.metric_values = []
        torch.backends.cudnn.benchmark = True

    def _init_device(self):
        device_config = self.config.get('device', {})
        cuda_device = device_config.get('cuda_device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(cuda_device)

    def _init_data_loaders(self):
        split_json = self.config['data']['split_json']

        datalist = load_decathlon_datalist(
            split_json, is_segmentation=False, data_list_key=self.config['data']['datasets_key']
        )
        validation_keys = self.config['data'].get('validation_keys')
        if validation_keys is None:
            validation_keys = {'validation': self.config['data']['validation_key']}

        from data.Augmentation import train_transforms, val_transforms

        train_ds = CacheDataset(
            data=datalist, transform=train_transforms,
            cache_num=self.config['training']['cache_num'],
            cache_rate=self.config['training']['cache_rate'],
            num_workers=self.config['training']['num_workers'],
        )
        val_loaders = {}
        for val_name, val_key in validation_keys.items():
            val_files = load_decathlon_datalist(
                split_json, is_segmentation=False, data_list_key=val_key
            )
            val_ds = CacheDataset(
                data=val_files, transform=val_transforms,
                cache_num=self.config['validation']['cache_num'],
                cache_rate=self.config['validation']['cache_rate'],
                num_workers=self.config['validation']['num_workers'],
            )
            val_loaders[val_name] = DataLoader(
                val_ds, batch_size=self.config['validation']['batch_size'], shuffle=False,
                num_workers=self.config['validation']['num_workers']
            )

        train_loader = DataLoader(train_ds, batch_size=self.config['training']['batch_size'], shuffle=True,
                                  num_workers=self.config['training']['num_workers'])
        return train_loader, val_loaders

    def _init_model(self):
        from models.getmodel import create_model
        return create_model()

    def _init_loss(self):
        # ================== 核心修改点：加入 FocalLoss 选项 ==================
        loss_type = self.config['loss'].get('loss_type', 'CrossEntropyLoss')

        if loss_type == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif loss_type == 'BCELoss':
            return torch.nn.BCELoss()
        elif loss_type == 'FocalLoss':
            # 自动从 yaml 中读取 gamma 和 alpha。如果没写，就用默认值
            gamma = self.config['loss'].get('gamma', 2.0)
            alpha = self.config['loss'].get('alpha', None)
            print(f"✅ 使用 FocalLoss (gamma={gamma}, alpha={alpha})")
            return FocalLoss(alpha=alpha, gamma=gamma)
        return None
        # =====================================================================

    def _init_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'],
                                weight_decay=self.config['training']['weight_decay'])

    def _init_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                          patience=self.config['scheduler']['patience'])
