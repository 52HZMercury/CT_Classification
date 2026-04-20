import sys
import os
from tqdm import tqdm  # 引入 tqdm 用于进度条

# 1. 获取项目根目录绝对路径 (CT_Classification)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. 强制把当前工作目录(CWD)切换到项目根目录！
os.chdir(project_root)

# 3. 把项目根目录加入导包路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import yaml
from monai.data import CacheDataset, load_decathlon_datalist
from torch.utils.data import DataLoader
from models.multimodel_resnet import MultiModelResNet
from data.Augmentation import val_transforms

# ==========================================
# 修改配置文件路径：使用绝对路径确保能找到
# ==========================================
config_path = os.path.join(project_root, "config/config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def load_model(checkpoint_path, device):
    """
    加载你训练好的多模态模型
    """
    model = MultiModelResNet(
        tabular_dim=config['model']['tabular_dim'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_data(dataloader, num_samples, device):
    images_list = []
    tabular_list = []

    for batch in dataloader:
        img = batch["image"].to(device)
        ca = batch["CA"].to(device)
        cac = batch["CAC"].to(device)
        x_image = torch.cat([img, ca, cac], dim=1)

        x_tabular = batch["tabular_features"]
        if isinstance(x_tabular, list):
            x_tabular = torch.stack(x_tabular, dim=1)
        x_tabular = x_tabular.to(device).float()

        images_list.append(x_image)
        tabular_list.append(x_tabular)

        if len(images_list) * images_list[0].shape[0] >= num_samples:
            break

    all_images = torch.cat(images_list, dim=0)[:num_samples]
    all_tabular = torch.cat(tabular_list, dim=0)[:num_samples]

    return all_images, all_tabular


def main():
    # 定义保存目录
    save_dir = "/workdir2/cn24/program/CT_Classification/logs/exp_260410-1558/visualization"
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    checkpoint_path = "/workdir2/cn24/program/CT_Classification/logs/exp_260410-1558/checkpoint/best_metric_model_0.8554.pth"

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model = load_model(checkpoint_path, device)

    dataset_json = config['data']['split_json']
    val_data_list = load_decathlon_datalist(dataset_json, is_segmentation=False, data_list_key="validation")

    val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("Preparing background data...")
    bg_images, bg_tabular = prepare_data(val_loader, num_samples=10, device=device)

    print("Preparing test data to explain...")
    test_images, test_tabular = prepare_data(val_loader, num_samples=5, device=device)

    print("Initializing SHAP Explainer...")
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    explainer = shap.GradientExplainer(model, [bg_images, bg_tabular], batch_size=1)

    print("Calculating SHAP values with progress bar...")
    num_test_samples = test_images.shape[0]

    # 初始化用于存储 SHAP 值的列表
    all_image_shap = []
    all_tabular_shap = []

    # 逐个样本计算 SHAP 值并显示进度条
    for i in tqdm(range(num_test_samples), desc="Calculating SHAP"):
        # 取出单个样本，保持其 batch_size=1 的维度
        single_image = test_images[i:i + 1]
        single_tabular = test_tabular[i:i + 1]

        shap_values = explainer.shap_values([single_image, single_tabular])

        # 解析输出格式
        if isinstance(shap_values, list) and len(shap_values) == 2:
            image_s = shap_values[1][0]
            tabular_s = shap_values[1][1]
        else:
            image_s = shap_values[0]
            tabular_s = shap_values[1]

        all_image_shap.append(image_s)
        all_tabular_shap.append(tabular_s)

    # 将所有样本的 SHAP 值拼接起来
    image_shap = np.concatenate(all_image_shap, axis=0)
    tabular_shap = np.concatenate(all_tabular_shap, axis=0)

    # ---------------------------------------------------------
    # 新增：将计算好的 SHAP 值保存为 .npy 文件
    # ---------------------------------------------------------
    image_shap_save_path = os.path.join(save_dir, "image_shap_values.npy")
    tabular_shap_save_path = os.path.join(save_dir, "tabular_shap_values.npy")

    np.save(image_shap_save_path, image_shap)
    np.save(tabular_shap_save_path, tabular_shap)
    print(f"Successfully saved SHAP values to:\n- {image_shap_save_path}\n- {tabular_shap_save_path}")

    # 5. 绘制临床数据的 SHAP 图 (Summary Plot)
    print("Plotting Tabular SHAP Summary...")
    tabular_features_np = test_tabular.cpu().numpy()

    feature_names = [
        'Male', 'Age', 'BMI', 'STS_score', 'Hypertension', 'Diabetes', 'COPD',
        'Coronary_artery_disease', 'Chronic_kidney_disease', 'Prior_atrial_fibrillation',
        'Peripheral_vascular_disease', 'Prior_stroke_TIA', 'Aortic_valve_calcification_volume',
        'calcified_raphe', 'Annulus_angulation', 'Annular_perimeter', 'Annular_area',
        'SOV_perimeter', 'STJ_diameter', 'Left_coronary_artery_ostium_height',
        'Right_coronary_artery_ostium_height', 'Maximal_diameter_of_ascending_aorta',
        'LVOT_perimeter', 'Aortic_regurgitation_moderate', 'Mean_aortic_valve_gradient',
        'Peak_aortic_valve_velocity', 'LVEF', 'LVEDD', 'IVS', 'THV_brand_1', 'THV_size',
        'Pre_dilatation', 'Post_dilatation', 'Implantation_of_multiple_THVs',
        'Peak_velocity_at_30_days', 'Mean_gradient_at_30_days', 'PVL_moderate_at_30_days'
    ]

    shap_summary_save_path = os.path.join(save_dir, "shap_summary_clinical.png")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(tabular_shap, tabular_features_np, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (Clinical Features)")
    plt.tight_layout()
    plt.savefig(shap_summary_save_path, dpi=300)
    plt.close()
    print(f"Saved clinical SHAP plot to {shap_summary_save_path}")

    # 6. (可选) 绘制 3D 图像的中间切片 SHAP 图
    print("Plotting Image SHAP (Center Slice)...")
    D_center = test_images.shape[4] // 2

    image_shap_sample_slice = image_shap[0, 0, :, :, D_center]
    raw_image_slice = test_images[0, 0, :, :, D_center].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_image_slice, cmap='gray')
    axes[0].set_title("Original CT (Center Slice)")
    axes[0].axis('off')

    max_val = np.max(np.abs(image_shap_sample_slice))
    axes[1].imshow(image_shap_sample_slice, cmap='bwr', vmin=-max_val, vmax=max_val)
    axes[1].set_title("SHAP Value Heatmap")
    axes[1].axis('off')

    shap_slice_save_path = os.path.join(save_dir, "SHAP_Slice.png")
    plt.savefig(shap_slice_save_path, dpi=300)
    plt.close()
    print(f"Saved image SHAP plot to {shap_slice_save_path}")


if __name__ == "__main__":
    main()