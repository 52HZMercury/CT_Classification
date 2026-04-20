import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import CacheDataset, load_decathlon_datalist

# 导入您项目中的模块
from models.multimodel_resnet import MultiModelResNet
from data.Augmentation import val_transforms

def main():
    # --- 配置路径 ---
    config_path = "config/config.yaml"
    model_weights = "/workdir2/cn24/program/CT_Classification/logs/exp_260328-1752/checkpoint/best_metric_model_0.7688.pth"
    save_dir = "/workdir2/cn24/program/CT_Classification/logs/exp_260328-1752/visualization"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device']['cuda_device'] if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型并加载权重
    model = MultiModelResNet(
        tabular_dim=config['model']['tabular_dim'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)

    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    print(f"✅ 模型权重已加载: {model_weights}")

    # 2. 准备验证集数据
    dataset_json = config['data']['split_json']
    val_data_list = load_decathlon_datalist(dataset_json, is_segmentation=False, data_list_key="validation")

    val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 3. 推理获取概率
    all_labels = []
    all_probs = []
    all_names = []  # <--- 新增：用于存储病人ID

    print("🚀 开始推理验证集...")
    with torch.no_grad():
        for batch in val_loader:
            # --- 提取病人 ID ---
            # MONAI 加载后的路径通常在 image_meta_dict 中
            raw_path = batch["image_meta_dict"]["filename_or_obj"][0]
            # 提取文件名（不带路径），然后取第一个点之前的部分
            patient_id = os.path.basename(raw_path).split('.')[0]
            all_names.append(patient_id)

            # 图像模态拼接 (Image + Pred)
            img = batch["image"].to(device)
            # pred = batch["pred"].to(device)
            # x_image = torch.cat([img, pred], dim=1)
            ca = batch["CA"].to(device)
            cac = batch["CAC"].to(device)
            x_image = torch.cat([img, ca, cac], dim=1)

            # 临床特征处理
            x_tabular = batch["tabular_features"]
            if isinstance(x_tabular, list):
                x_tabular = torch.stack(x_tabular, dim=1)
            x_tabular = x_tabular.to(device).float()

            labels = batch["label"].to(device)

            # 前向传播
            logits = model(x_image, x_tabular)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. 保存数据为 numpy 格式
    output_file = os.path.join(save_dir, "eval_data.npz")
    # 将 ID 也存入 npz 文件
    np.savez(
        output_file,
        y_true=np.array(all_labels),
        y_probs=np.array(all_probs),
        patient_ids=np.array(all_names)  # <--- 新增保存项
    )
    print(f"💾 推理完成！共处理 {len(all_names)} 个样本。")
    print(f"📂 数据已保存至: {output_file}")


if __name__ == "__main__":
    main()