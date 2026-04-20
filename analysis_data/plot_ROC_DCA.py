import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def calculate_net_benefit(y_true, y_probs, thresholds):
    """计算 DCA 的 Net Benefit"""
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if thresh == 1.0:
            net_benefit = 0
        else:
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)
    return net_benefits


def main():
    # --- 配置路径 ---
    data_path = "/workdir1.8t/cn24/program/CT_Classification/logs/exp_260315-0512//visualization/eval_data.npz"
    save_dir = "/workdir1.8t/cn24/program/CT_Classification/logs/exp_260315-0512/visualization"

    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到数据文件 {data_path}，请先确保已运行 inference.py")
        return

    # 1. 加载数据
    data = np.load(data_path)
    y_true = data['y_true']
    y_probs = data['y_probs']

    # ==========================================
    # 2. 绘制并保存 ROC 曲线
    # ==========================================
    plt.figure(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Model (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    roc_save_path = os.path.join(save_dir, "ROC_Curve.png")
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前画布
    print(f"✅ ROC 曲线已保存至: {roc_save_path}")

    # ==========================================
    # 3. 绘制并保存 DCA 曲线
    # ==========================================
    plt.figure(figsize=(6, 6))
    thresholds = np.linspace(0, 0.99, 100)
    model_nb = calculate_net_benefit(y_true, y_probs, thresholds)

    # 基准线计算
    none_nb = [0] * len(thresholds)
    prevalence = np.mean(y_true)
    all_nb = [(prevalence - (1 - prevalence) * (t / (1 - t))) if t < 1 else 0 for t in thresholds]

    plt.plot(thresholds, model_nb, color='#d62728', lw=2.5, label='Model')
    plt.plot(thresholds, all_nb, color='black', lw=1.5, linestyle='--', label='Treat All')
    plt.plot(thresholds, none_nb, color='blue', lw=1.5, label='Treat None')

    plt.ylim([-0.05, prevalence + 0.15])
    plt.xlim([0, 1.0])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis (DCA)', fontsize=13, fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    dca_save_path = os.path.join(save_dir, "DCA_Curve.png")
    plt.savefig(dca_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ DCA 曲线已保存至: {dca_save_path}")


if __name__ == "__main__":
    main()