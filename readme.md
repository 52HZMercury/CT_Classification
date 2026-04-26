# CT Classification

基于 CT 影像和结构化特征的二分类训练项目。项目使用 PyTorch/MONAI 构建数据流程、模型训练、验证评估和推理结果导出。

## 项目结构

```text
.
├── config/              # 训练、模型、数据和预处理配置
├── data/                # 数据增强与预处理
├── metadata/            # 数据集划分 JSON
├── models/              # 模型定义
├── utils/               # 数据处理工具
├── main.py              # 训练入口
├── inference_pvt.py     # 推理入口
├── train.py             # 单轮训练逻辑
├── trainer.py           # 训练器与评估流程
└── validation.py        # 验证指标计算
```

## 配置

训练参数统一维护在 `config/config.yaml`，常用字段包括：

- `data.split_json`：数据集划分文件
- `data.out_dir`：日志和模型输出目录
- `model.architecture`：模型结构
- `model.in_channels`：输入通道数
- `model.tabular_dim`：结构化特征维度
- `training.max_epochs`：最大训练轮数
- `training.batch_size`：训练批大小
- `device.cuda_device`：训练设备

## 训练

前台运行：

```bash
python main.py
```

后台运行：

```bash
nohup python main.py >> output.log 2>&1 &
```

训练输出默认保存到 `data.out_dir`，实验名会自动生成为 `exp_YYMMDD-HHMM`。

## 推理

先在 `inference_pvt.py` 中确认模型权重路径和结果保存目录：

```python
model_weights = "path/to/best_metric_model.pth"
save_dir = "path/to/visualization"
```

然后运行：

```bash
python inference_pvt.py
```

推理结果会保存为 `eval_data.npz`，包含真实标签、预测概率和样本 ID。

## 代理设置（可选）

如需配置网络代理：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```