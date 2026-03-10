"""
使用epoch控制训练过程的主训练文件 (分类版本)
"""
import os
import tempfile
import yaml
from datetime import datetime  # 新增：引入 datetime
from train import train
from trainer import Trainer

import warnings
warnings.filterwarnings("ignore")

# 加载配置
config_path = "config/config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ==========================================
# 动态生成实验名称: exp_YYMMDD-HHMM
# ==========================================
current_time = datetime.now().strftime("%y%m%d-%H%M")
config['data']['exp_name'] = f"exp_{current_time}"
# ==========================================

def main():
    # 注意：现在我们需要把修改后的 config 对象传给 Trainer，
    # 而不是只传 config_path 重新读取一遍。
    trainer = Trainer(config=config)

    # 设置输出目录
    out_dir = config['data']['out_dir']
    root_dir = tempfile.mkdtemp() if out_dir is None else out_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # 训练参数
    max_epochs = config['training']['max_epochs']
    early_stop_counter = 0
    current_epoch = 0

    # 修改：初始化分类相关的指标
    acc_val_best = 0.0
    auc_val_best = 0.0
    global_step_best = 0

    print(f"开始分类模型训练 ---------------------------------------------------")
    print(f"模型：ResNet18-3D | 类别数：{config['model']['num_classes']}")
    print(f"输出目录：{os.path.abspath(root_dir)}")

    # 训练循环
    for epoch in range(max_epochs):
        current_epoch += 1

        # 调用修改后的 train 函数
        (current_epoch,
         acc_val_best,
         auc_val_best,
         global_step_best,
         early_stop_counter,
         stop_now) = train(
            trainer,
            current_epoch,
            acc_val_best,
            auc_val_best,
            global_step_best,
            early_stop_counter,
            config
        )

        if stop_now:
            print(f"\n[早停] 训练在第 {current_epoch} 轮停止。")
            break

    print("")
    print(f"======================= 训练完成 ========================")
    print(f"== 实验名称  : {config['data']['exp_name']}")
    print(f"== 最佳准确率 (Acc): {acc_val_best:.4f}")
    print(f"== 对应 AUC    : {auc_val_best:.4f}")
    print(f"== 最佳轮次 (Epoch): {global_step_best}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()