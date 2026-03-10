import yaml
import torch
import os
from torch.utils.tensorboard import SummaryWriter

# 【修改点 1】：将 AccuracyMetric 替换为 ConfusionMatrixMetric
from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist
from monai.transforms import AsDiscrete
from monai.networks.utils import one_hot

class Trainer:
    def __init__(self, config):  # 修改：直接接收配置字典
        """
        初始化训练器
        """
        self.config = config  # 直接赋值

        self.device = self._init_device()
        self.train_loader, self.val_loader = self._init_data_loaders()
        self.model = self._init_model().to(self.device)

        # 分类专用损失函数
        self.loss_function = self._init_loss()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        log_dir = os.path.join(self.config['data']['out_dir'], self.config['data']['exp_name'])
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # --- 分类专用指标 ---
        # 用于将模型输出转换为 one-hot 或 argmax
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.config['model']['num_classes'])
        self.post_label = AsDiscrete(to_onehot=self.config['model']['num_classes'])

        # 【修改点 2】：使用 ConfusionMatrixMetric 计算 Accuracy
        # include_background=True 表示所有类别（包括索引为0的类）都参与分类计算
        self.metric = ConfusionMatrixMetric(
            metric_name=["accuracy", "precision", "recall", "specificity", "negative_predictive_value"],
            compute_sample=True,
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

        # 注意：is_segmentation=False, 因为 label 是数值而非图像文件
        datalist = load_decathlon_datalist(
            split_json, is_segmentation=False, data_list_key=self.config['data']['datasets_key']
        )
        val_files = load_decathlon_datalist(
            split_json, is_segmentation=False, data_list_key=self.config['data']['validation_key']
        )

        from data.Augmentation import train_transforms, val_transforms

        train_ds = CacheDataset(
            data=datalist, transform=train_transforms,
            cache_num=self.config['training']['cache_num'],
            cache_rate=self.config['training']['cache_rate'],
            num_workers=self.config['training']['num_workers'],
        )
        val_ds = CacheDataset(
            data=val_files, transform=val_transforms,
            cache_num=self.config['validation']['cache_num'],
            cache_rate=self.config['validation']['cache_rate'],
            num_workers=self.config['validation']['num_workers'],
        )

        train_loader = DataLoader(train_ds, batch_size=self.config['training']['batch_size'], shuffle=True,
                                  num_workers=self.config['training']['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=self.config['validation']['batch_size'], shuffle=False,
                                num_workers=self.config['validation']['num_workers'])

        return train_loader, val_loader

    def _init_model(self):
        from models.getmodel import create_model
        return create_model()

    def _init_loss(self):
        if self.config['loss']['loss_type'] == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif self.config['loss']['loss_type'] == 'BCELoss':
            return torch.nn.BCELoss()
        return None

    def _init_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'],
                                weight_decay=self.config['training']['weight_decay'])

    def _init_scheduler(self):
        # 分类通常监控准确率，准确率越高越好，所以用 'max'
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                          patience=self.config['scheduler']['patience'])