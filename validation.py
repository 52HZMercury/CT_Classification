import yaml
import torch
from monai.networks.utils import one_hot

config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def validation(Trainer):
    Trainer.model.eval()
    Trainer.metric.reset()
    Trainer.auc_metric.reset()

    with torch.no_grad():
        for batch_data in Trainer.val_loader:
            val_images, val_labels = (
                batch_data["image"].to(Trainer.device),
                batch_data["label"].to(Trainer.device),
            )
            logits = Trainer.model(val_images)

            # 计算预测类别 (Argmax)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            # 计算概率 (Softmax) 用于 AUC
            probs = torch.softmax(logits, dim=1)

            # 转换为 one-hot 格式以匹配指标计算要求
            n_classes = config['model']['num_classes']
            val_labels_oh = one_hot(val_labels.unsqueeze(1), num_classes=n_classes)
            preds_oh = one_hot(preds, num_classes=n_classes)

            Trainer.metric(y_pred=preds_oh, y=val_labels_oh)
            Trainer.auc_metric(y_pred=probs, y=val_labels_oh)

        # ================== 核心修改点 ==================
        # aggregate() 现在返回一个长度为 4 的 list
        metric_results = Trainer.metric.aggregate()

        # 按照 ["accuracy", "precision", "recall", "specificity"] 的顺序解包
        # 使用 .mean().item() 将多分类的张量转化为单个标量数值
        # 按照 ["accuracy", "precision", "recall", "specificity", "negative_predictive_value"] 的顺序解包
        acc_val = metric_results[0].mean().item()
        ppv_val = metric_results[1].mean().item()  # Precision 就是 PPV
        rec_val = metric_results[2].mean().item()
        spec_val = metric_results[3].mean().item()
        npv_val = metric_results[4].mean().item()  # 取出 NPV

        auc_val = Trainer.auc_metric.aggregate().item()

    # 将原来返回的 prec_val 改名为 ppv_val，并增加 npv_val
    return acc_val, auc_val, ppv_val, npv_val, rec_val, spec_val