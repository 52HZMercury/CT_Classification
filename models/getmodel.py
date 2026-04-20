"""
Model initialization module based on config file
"""
import os
import sys
import yaml
import torch
from models.resnet import resnet18_3d
from models.multimodel_resnet import MultiModelResNet

# --- Numpy 兼容性处理 ---
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath


config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def create_model():
    """
    Create model based on config parameters

    Returns:
        torch.nn.Module: Initialized model
    """
    architecture = config['model'].get('architecture', 'resnet18')

    if architecture.lower() == 'resnet18':
        model = resnet18_3d(config['model']['in_channels'], config['model']['num_classes'])
    elif architecture.lower() == 'multimodelresnet':
        model = MultiModelResNet(tabular_dim=config['model']['tabular_dim'], in_channels=config['model']['in_channels'], num_classes=config['model']['num_classes'])

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model