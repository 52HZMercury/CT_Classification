import torch
from monai.networks.utils import one_hot

def validation(Trainer, val_loader=None):
    Trainer.model.eval()
    Trainer.auc_metric.reset()
    if val_loader is None:
        val_loader = Trainer.val_loader

    # 用于手动收集整个验证集的所有预测和真实标签
    all_preds = []
    all_labels = []


    with torch.no_grad():
        for batch in val_loader:
            # 1. 同样取出三个模态
            img = batch["image"].to(Trainer.device)
            # ca = batch["CA"].to(Trainer.device)
            cac = batch["CAC"].to(Trainer.device)
            # 2. 在通道维度拼接
            val_images = torch.cat([img, cac], dim=1)  # 拼接后变为 [B, 3, H, W, D]

            # pred = batch["pred"].to(Trainer.device)
            # val_images = torch.cat([img, pred], dim=1)


            val_tabular = batch["tabular_features"]
            if isinstance(val_tabular, list):
                val_tabular = torch.stack(val_tabular, dim=1)  # 处理可能的格式转换
            val_tabular = val_tabular.to(Trainer.device).float()

            val_labels = batch["label"].to(Trainer.device)

            # 3. 传入模型
            logits = Trainer.model(val_images, val_tabular)
            # logits = Trainer.model(val_images)

            # 1. 计算预测类别 (Argmax) 和概率 (Softmax)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            # 2. 收集结果，存入列表
            all_preds.append(preds.cpu())
            all_labels.append(val_labels.cpu())

            # 3. AUC 指标不受影响，继续用 MONAI 的计算
            n_classes = Trainer.config['model']['num_classes']
            val_labels_oh = one_hot(val_labels.unsqueeze(1), num_classes=n_classes)
            Trainer.auc_metric(y_pred=probs, y=val_labels_oh)

    # ================= 核心修改点：纯 PyTorch 手动算指标 =================
    # 将所有的 batch 拼接成一个长长的一维张量
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 统计全局的 TP, TN, FP, FN (针对阳性类 1)
    TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
    TN = ((all_preds == 0) & (all_labels == 0)).sum().item()
    FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
    FN = ((all_preds == 0) & (all_labels == 1)).sum().item()

    # 计算各项医学指标 (加上 1e-8 防止模型早期全猜阴性导致除以 0 报错)
    acc_val  = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    ppv_val  = TP / (TP + FP + 1e-8)  # 阳性预测值 (Precision)
    npv_val  = TN / (TN + FN + 1e-8)  # 阴性预测值
    rec_val  = TP / (TP + FN + 1e-8)  # 召回率/敏感度 (Sensitivity)
    spec_val = TN / (TN + FP + 1e-8)  # 特异度 (Specificity)

    # 提取 AUC
    auc_val = Trainer.auc_metric.aggregate().item()
    # =================================================================

    return acc_val, auc_val, ppv_val, npv_val, rec_val, spec_val
