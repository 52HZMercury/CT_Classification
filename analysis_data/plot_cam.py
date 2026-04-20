import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from monai.data import CacheDataset, load_decathlon_datalist
from torch.utils.data import DataLoader

# 1. 动态添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你的模型和数据处理
from models.multimodel_resnet import MultiModelResNet
from data.Augmentation import val_transforms


# ==========================================
# 核心类：自定义 3D Grad-CAM
# ==========================================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册前向和反向 Hook，用于截获特征图和梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x_image, x_tabular, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # 1. 前向传播
        logits = self.model(x_image, x_tabular)

        # 如果没有指定目标类别，则默认解释模型预测得分最高的那个类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # 2. 反向传播，获取目标类别的梯度
        score = logits[:, target_class]
        score.backward(retain_graph=True)

        # 3. 计算 Grad-CAM
        # 获取截获的梯度和特征图 (形状通常为 [B, C, H, W, D])
        b, k, *spatial_dims = self.gradients.size()

        # 对空间维度求全局平均池化 (GAP)，得到每个通道的权重 alpha
        # 3D 数据的空间维度被展平后求均值
        alpha = self.gradients.view(b, k, -1).mean(2)  # shape: [B, C]
        alpha = alpha.view(b, k, *([1] * len(spatial_dims)))  # shape: [B, C, 1, 1, 1]

        # 用 alpha 对特征图进行加权求和
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)  # shape: [B, 1, H', W', D']

        # ReLU 激活，只保留对正向预测有贡献的特征
        cam = F.relu(cam)

        # 4. 将 CAM 插值放大回原始图像的尺寸
        cam = F.interpolate(cam, size=x_image.shape[2:], mode='trilinear', align_corners=False)

        # 归一化到 [0, 1] 区间以便可视化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), target_class, logits.detach().cpu().numpy()


# ==========================================
# 辅助函数：模型加载与数据准备
# ==========================================
config_path = os.path.join(project_root, "config/config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_model(checkpoint_path, device):
    model = MultiModelResNet(
        tabular_dim=config['model']['tabular_dim'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def main():
    save_dir = os.path.join(project_root, "logs/exp_260410-1558/visualization")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(project_root, "logs/exp_260410-1558/checkpoint/best_metric_model_0.8554.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = load_model(checkpoint_path, device)

    # 【重要提示】：你需要指定目标卷积层！
    # 通常是 ResNet 的最后一个卷积层。如果你的模型使用了 monai 的 resnet，通常叫 model.resnet.layer4[-1]
    # 如果运行报错说找不到 layer4，请执行 print(model) 查看你的模型结构并修改下方代码
    try:
        target_layer = model.resnet.layer4[-1]
    except AttributeError:
        print("尝试寻找 target_layer 失败，请查看模型结构并手动指定！")
        print(model)
        return

    # 初始化自定义的 3D Grad-CAM
    grad_cam = GradCAM3D(model, target_layer)

    # 加载一条测试数据
    dataset_json = config['data']['split_json']
    val_data_list = load_decathlon_datalist(dataset_json, is_segmentation=False, data_list_key="validation")
    val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("Running Grad-CAM...")
    n = 4  # 你想取的 batch 序号
    it = iter(val_loader)
    for _ in range(n):
        batch = next(it)

    img = batch["image"].to(device)
    ca = batch["CA"].to(device)
    cac = batch["CAC"].to(device)
    x_image = torch.cat([img, ca, cac], dim=1)  # [B, 3, H, W, D]

    x_tabular = batch["tabular_features"]
    if isinstance(x_tabular, list):
        x_tabular = torch.stack(x_tabular, dim=1)
    x_tabular = x_tabular.to(device).float()

    # 生成 3D Grad-CAM
    cam_3d, predicted_class, logits = grad_cam.generate(x_image, x_tabular)
    print(f"Prediction Logits: {logits}, Target Class for CAM: {predicted_class}")

    # ==========================================
    # 5. 循环遍历所有深度切片并分别保存
    # ==========================================
    cam_volume = cam_3d[0, 0]  # 形状 [H, W, D]
    original_volume = x_image[0, 0].cpu().numpy()  # 形状 [H, W, D]

    depth = original_volume.shape[2]

    # 创建专门存放切片的子目录
    slices_dir = os.path.join(save_dir, "all_slices")
    os.makedirs(slices_dir, exist_ok=True)

    print(f"Total slices to save: {depth}. Processing...")

    for i in range(depth):
        orig_slice = original_volume[:, :, i]
        cam_slice = cam_volume[:, :, i]

        # 创建画布 (保持 1x3 布局)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 原始 CT 切片
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title(f"Original CT (Slice {i})")
        axes[0].axis('off')

        # 2. 纯热力图 (使用 vmin/vmax 确保所有切片的颜色量程一致)
        axes[1].imshow(cam_slice, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM Heatmap")
        axes[1].axis('off')

        # 3. 叠加图 (Overlay)
        axes[2].imshow(orig_slice, cmap='gray')
        axes[2].imshow(cam_slice, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f"Overlay (Slice {i})")
        axes[2].axis('off')

        # 保存当前切片，使用 :03d 格式化文件名以便在文件夹内按顺序排序
        slice_save_path = os.path.join(slices_dir, f"GradCAM_Slice_{i:03d}.png")
        plt.savefig(slice_save_path, dpi=200, bbox_inches='tight')

        # 【重要】必须执行 plt.close()，否则会内存泄漏（内存被几百张图堆满）
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{depth} slices saved.")

    print(f"\nSuccessfully saved all {depth} Grad-CAM slices to: {slices_dir}")


if __name__ == "__main__":
    main()