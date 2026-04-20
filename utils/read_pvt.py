import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def generate_dataset_json(input_file, output_dir='../metadata', output_filename='img_30daysSuccess.json'):
    """
    读取数据并生成 JSON，包含 image, CA, CAC 三路图像路径。
    """
    # 1. 检查并创建 metadata 目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    # 2. 加载文件
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, dtype={'ID': str})
    else:
        df = pd.read_excel(input_file, sheet_name="data_external_final", dtype={'ID': str})

    # 3. 路径配置
    base_data_dir = "/workdir2/cn24/data/30daysSuccess"
    # 定义子文件夹
    img_dir = os.path.join(base_data_dir, "image")
    ca_dir = os.path.join(base_data_dir, "CA")
    cac_dir = os.path.join(base_data_dir, "CAC")

    # 4. 准备数据列表
    data_list = []

    # 清洗数据：移除 ID 或 Label 为空的行
    df = df.dropna(subset=['ID', 'Device_success_at_30_days'])

    for _, row in df.iterrows():
        # 处理 ID (转换为字符串并去除前后空格)
        patient_id = str(row['ID']).split('.')[0].strip()

        # 映射标签：Yes -> 1, No -> 0
        raw_label = str(row['Device_success_at_30_days']).strip().lower()
        label_val = 1 if raw_label == 'yes' else 0

        # 按照要求格式构建字典，增加 CA 和 CAC 路径
        entry = {
            "image": os.path.join(img_dir, f"{patient_id}.nii.gz"),
            "CA": os.path.join(ca_dir, f"{patient_id}.nii.gz"),
            "CAC": os.path.join(cac_dir, f"{patient_id}.nii.gz"),
            "label": label_val
        }
        data_list.append(entry)

    # # 5. 划分数据集 (80% 训练, 20% 验证)
    # train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

    # 6. 划分数据集 (确保阳性样本的 80% 进入训练集，且整体比例 8:2)
    # 将正负样本分开
    pos_samples = [d for d in data_list if d['label'] == 1]
    neg_samples = [d for d in data_list if d['label'] == 0]

    # # 只要175例
    # neg_samples = neg_samples[:175]

    # 分别对正负样本进行 8:2 划分
    train_pos, val_pos = train_test_split(pos_samples, test_size=0.2, random_state=42)
    train_neg, val_neg = train_test_split(neg_samples, test_size=0.2, random_state=42)

    # 合并训练集和验证集
    train_data = train_pos + train_neg
    val_data = val_pos + val_neg

    # 打乱顺序，防止训练时模型先只看 1 再只看 0
    import random
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)

    # 6. 构建最终 JSON 结构
    final_output = {
        "training": train_data,
        "validation": val_data
    }

    # 7. 写入文件
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 成功！JSON 文件已保存至: {output_path}")
    print(f"📊 总样本数: {len(data_list)}")
    print(f"   - 阳性(1): {len(pos_samples)} | 阴性(0): {len(neg_samples)}")
    print(f"📊 训练集: {len(train_data)} (包含 {len(train_pos)} 个阳性)")
    print(f"📊 验证集: {len(val_data)} (包含 {len(val_pos)} 个阳性)")
    print("-" * 30)


if __name__ == "__main__":
    input_xlsx = '/workdir2/cn24/data/30daysSuccess/data_30success.xlsx'

    if os.path.exists(input_xlsx):
        generate_dataset_json(input_xlsx)
    else:
        print(f"❌ 找不到输入文件: {input_xlsx}")