import sys
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
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
import argparse
import gc
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


class SingleClassModel(torch.nn.Module):
    def __init__(self, model, class_idx=1):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, img, tab):
        logits = self.model(img, tab)
        return logits[:, self.class_idx:self.class_idx + 1]


def cleanup_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def prepare_data(dataloader, num_samples, device, skip_samples=0):
    images_list = []
    tabular_list = []
    seen_samples = 0

    for batch in dataloader:
        img = batch["image"].to(device)
        # ca = batch["CA"].to(device)
        cac = batch["CAC"].to(device)
        # x_image = torch.cat([img, ca, cac], dim=1)
        x_image = torch.cat([img, cac], dim=1)

        x_tabular = batch["tabular_features"]
        if isinstance(x_tabular, list):
            x_tabular = torch.stack(x_tabular, dim=1)
        x_tabular = x_tabular.to(device).float()

        batch_size = x_image.shape[0]
        if seen_samples + batch_size <= skip_samples:
            seen_samples += batch_size
            continue
        if seen_samples < skip_samples:
            start = skip_samples - seen_samples
            x_image = x_image[start:]
            x_tabular = x_tabular[start:]
        seen_samples += batch_size

        images_list.append(x_image)
        tabular_list.append(x_tabular)

        if len(images_list) * images_list[0].shape[0] >= num_samples:
            break

    if not images_list:
        raise ValueError(f"No samples collected. skip_samples={skip_samples} may be too large for this dataloader.")

    all_images = torch.cat(images_list, dim=0)[:num_samples]
    all_tabular = torch.cat(tabular_list, dim=0)[:num_samples]

    return all_images, all_tabular


def parse_args():
    parser = argparse.ArgumentParser(description="Low-memory SHAP plotting for multimodal CT model.")
    parser.add_argument("--background-samples", type=int, default=1,
                        help="Number of background samples. Keep this small for 3D CT on 24GB GPUs.")
    parser.add_argument("--test-samples", type=int, default=5,
                        help="Number of validation samples to explain.")
    parser.add_argument("--test-skip-samples", type=int, default=None,
                        help="Samples to skip before selecting test cases. Default: background-samples.")
    parser.add_argument("--shap-nsamples", type=int, default=16,
                        help="Number of SHAP integration samples per explained case.")
    parser.add_argument("--class-idx", type=int, default=1,
                        help="Class index to explain. Default 1 is the positive class.")
    parser.add_argument("--image-channel", type=int, default=0,
                        help="Image channel to visualize. 0 is CT, 1 is CAC for the current input.")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="Depth slice index to visualize. Default: slice with strongest absolute SHAP.")
    parser.add_argument("--plot-sample-index", type=int, default=None,
                        help="Explained sample index to visualize. Default: sample with strongest absolute image SHAP.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Torch device, e.g. cuda:0, cuda:3, or cpu.")
    parser.add_argument("--save-dir",
                        default="/workdir2/cn24/program/CT_Classification/logs/exp_260427-0029/visualization")
    parser.add_argument("--checkpoint-path",
                        default="/workdir2/cn24/program/CT_Classification/logs/exp_260427-0029/checkpoint/best_metric_model_0.9444.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    # 定义保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    checkpoint_path = args.checkpoint_path

    device = torch.device(args.device)
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    model = SingleClassModel(model, class_idx=args.class_idx).to(device).eval()

    dataset_json = config['data']['split_json']
    val_data_list = load_decathlon_datalist(dataset_json, is_segmentation=False, data_list_key="validation_huaxi")

    val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("Preparing background data...")
    bg_images, bg_tabular = prepare_data(val_loader, num_samples=args.background_samples, device=torch.device("cpu"))
    bg_images = bg_images.to(device)
    bg_tabular = bg_tabular.to(device)

    print("Preparing test data to explain...")
    test_skip_samples = args.background_samples if args.test_skip_samples is None else args.test_skip_samples
    test_images, test_tabular = prepare_data(
        val_loader,
        num_samples=args.test_samples,
        device=torch.device("cpu"),
        skip_samples=test_skip_samples
    )
    print(f"Using {args.background_samples} background sample(s); skipped {test_skip_samples} sample(s) for test data.")

    print("Initializing SHAP Explainer...")
    cleanup_cuda(device)

    explainer = shap.GradientExplainer(model, [bg_images, bg_tabular], batch_size=1)
    cleanup_cuda(device)

    print("Calculating SHAP values with progress bar...")
    num_test_samples = test_images.shape[0]

    # 初始化用于存储 SHAP 值的列表
    all_image_shap = []
    all_tabular_shap = []

    # 逐个样本计算 SHAP 值并显示进度条
    for i in tqdm(range(num_test_samples), desc="Calculating SHAP"):
        # 取出单个样本，保持其 batch_size=1 的维度
        single_image = test_images[i:i + 1].to(device)
        single_tabular = test_tabular[i:i + 1].to(device)

        shap_values = explainer.shap_values([single_image, single_tabular], nsamples=args.shap_nsamples)

        # 解析输出格式
        if isinstance(shap_values, list) and len(shap_values) == 1 and isinstance(shap_values[0], list):
            image_s = shap_values[0][0]
            tabular_s = shap_values[0][1]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            image_s = shap_values[0]
            tabular_s = shap_values[1]
        else:
            image_s = shap_values[0]
            tabular_s = shap_values[1]

        all_image_shap.append(image_s)
        all_tabular_shap.append(tabular_s)
        del single_image, single_tabular, shap_values
        cleanup_cuda(device)

    # 将所有样本的 SHAP 值拼接起来
    image_shap = np.concatenate(all_image_shap, axis=0)
    tabular_shap = np.concatenate(all_tabular_shap, axis=0)

    # ---------------------------------------------------------
    # 新增：将计算好的 SHAP 值保存为 .npy 文件
    # ---------------------------------------------------------
    image_shap_save_path = os.path.join(save_dir, "image_shap_values_huaxi.npy")
    tabular_shap_save_path = os.path.join(save_dir, "tabular_shap_values_huaxi.npy")

    np.save(image_shap_save_path, image_shap)
    np.save(tabular_shap_save_path, tabular_shap)
    print(f"Successfully saved SHAP values to:\n- {image_shap_save_path}\n- {tabular_shap_save_path}")
    image_abs_scores = np.mean(np.abs(image_shap), axis=tuple(range(1, image_shap.ndim)))
    print("Mean absolute image SHAP per explained sample:", image_abs_scores)

    # 5. 绘制临床数据的 SHAP 图 (Summary Plot)
    print("Plotting Tabular SHAP Summary...")
    tabular_features_np = test_tabular.cpu().numpy()

    feature_names = [
        'Male', 'Age', 'BMI', 'STS_score', 'Hypertension', 'Diabetes', 'Coronary_artery_disease',
        'Chronic_kidney_disease', 'Prior_atrial_fibrillation', 'Prior_stroke_TIA',
        'Aortic_valve_calcification_volume', 'calcified_raphe', 'Annulus_angulation', 'Annular_perimeter',
        'Annular_area', 'STJ_diameter', 'Left_coronary_artery_ostium_height',
        'Right_coronary_artery_ostium_height', 'LVOT_perimeter',
         'Mean_aortic_valve_gradient', 'Peak_aortic_valve_velocity',
        'LVEF', 'LVEDD', 'IVS'
    ]

    shap_summary_save_path = os.path.join(save_dir, "shap_summary_clinical_huaxi.png")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(tabular_shap, tabular_features_np, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (Clinical Features)")
    plt.tight_layout()
    plt.savefig(shap_summary_save_path, dpi=300)
    plt.close()
    print(f"Saved clinical SHAP plot to {shap_summary_save_path}")

    # 6. (可选) 绘制 3D 图像的中间切片 SHAP 图
    print("Plotting Image SHAP Slice...")
    if args.plot_sample_index is None:
        plot_sample_index = int(np.argmax(image_abs_scores))
    else:
        plot_sample_index = int(np.clip(args.plot_sample_index, 0, image_shap.shape[0] - 1))
    image_channel = min(args.image_channel, image_shap.shape[1] - 1)
    sample_image_shap = np.asarray(image_shap[plot_sample_index, image_channel])
    raw_image = test_images[plot_sample_index, image_channel].cpu().numpy()

    if sample_image_shap.ndim == 4 and sample_image_shap.shape[-1] == 1:
        sample_image_shap = sample_image_shap[..., 0]
    if sample_image_shap.shape != raw_image.shape:
        raise ValueError(
            f"SHAP/image shape mismatch: shap={sample_image_shap.shape}, image={raw_image.shape}. "
            "Please check SHAP output dimensions."
        )

    depth_size = raw_image.shape[2]
    if args.slice_index is None:
        slice_scores = np.mean(np.abs(sample_image_shap), axis=(0, 1))
        slice_index = int(np.argmax(slice_scores))
    else:
        slice_index = int(np.clip(args.slice_index, 0, depth_size - 1))

    image_shap_sample_slice = sample_image_shap[:, :, slice_index]
    raw_image_slice = raw_image[:, :, slice_index]
    abs_limit = np.percentile(np.abs(image_shap_sample_slice), 99)
    if not np.isfinite(abs_limit) or abs_limit == 0:
        abs_limit = np.max(np.abs(image_shap_sample_slice))
    if not np.isfinite(abs_limit) or abs_limit == 0:
        abs_limit = 1.0

    print(
        f"Visualizing sample={plot_sample_index}, channel={image_channel}, slice={slice_index}/{depth_size - 1}, "
        f"shap min={image_shap_sample_slice.min():.4e}, "
        f"max={image_shap_sample_slice.max():.4e}, abs_limit={abs_limit:.4e}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_image_slice, cmap='gray')
    axes[0].set_title(f"Original CT (Slice {slice_index})")
    axes[0].axis('off')

    axes[1].imshow(raw_image_slice, cmap='gray')
    overlay = axes[1].imshow(
        image_shap_sample_slice,
        cmap='bwr',
        vmin=-abs_limit,
        vmax=abs_limit,
        alpha=0.45
    )
    axes[1].set_title("SHAP Overlay")
    axes[1].axis('off')

    axes[2].imshow(image_shap_sample_slice, cmap='bwr', vmin=-abs_limit, vmax=abs_limit)
    axes[2].set_title("SHAP Heatmap")
    axes[2].axis('off')
    fig.colorbar(overlay, ax=axes, fraction=0.025, pad=0.02)

    shap_slice_save_path = os.path.join(save_dir, "SHAP_Slice_huaxi.png")
    plt.savefig(shap_slice_save_path, dpi=300)
    plt.close()
    print(f"Saved image SHAP plot to {shap_slice_save_path}")


if __name__ == "__main__":
    main()
