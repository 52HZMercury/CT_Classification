import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def generate_dataset_json(input_file, output_dir='../metadata', output_filename='pvt.json'):
    """
    读取数据并生成 JSON，保存到指定的 metadata 路径下。
    """
    # 1. 检查并创建 metadata 目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    # 2. 加载文件
    # 【核心修改点】：加入 dtype={'ID': str}，强制将 ID 列作为字符串读取，从而保留前置 0
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, dtype={'DicomID': str})
    else:
        df = pd.read_excel(input_file, dtype={'DicomID': str})

    # 3. 准备数据列表
    data_list = []
    img_prefix = "/workdir2/cn24/data/PVT_huaxi_pro/image/"

    # 清洗数据：移除 ID 或 Label 为空的行
    df = df.dropna(subset=['DicomID', 'PVL_NotImproved_afterPostdilatation'])

    for _, row in df.iterrows():
        # 处理 ID (转换为字符串并去除前后空格)
        # 因为读取时已经强制为字符串，这里前置0会被完美保留
        # split('.')[0] 是为了兼容如果 Excel 里真的有人手动输入了 "0012.0" 这种奇葩文本
        patient_id = str(row['DicomID']).split('.')[0].strip()

        # 映射标签：Yes -> 1, No -> 0
        raw_label = str(row['PVL_NotImproved_afterPostdilatation']).strip().lower()
        label_val = 1 if raw_label == 'yes' else 0

        # 按照要求格式构建字典
        entry = {
            "image": f"{img_prefix}{patient_id}.nii.gz",
            "label": label_val
        }
        data_list.append(entry)

    # 4. 划分数据集 (80% 训练, 20% 验证)
    # random_state=42 用于确保每次运行结果一致，生产环境可根据需要修改
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

    # 5. 构建最终 JSON 结构
    final_output = {
        "training": train_data,
        "validation": val_data
    }

    # 6. 写入到 metadata/dataset.json
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 成功！JSON 文件已保存至: {output_path}")
    print(f"📊 总样本数: {len(data_list)}")
    print(f"📊 训练集数量: {len(train_data)}")
    print(f"📊 验证集数量: {len(val_data)}")
    print("-" * 30)


# ==========================================
# 主函数入口
# ==========================================
if __name__ == "__main__":
    # 请根据实际文件名修改此处
    input_csv = '/workdir2/cn24/data/PVT_huaxi_pro/data_PVL_V0309.xlsx'

    # 检查文件是否存在
    if os.path.exists(input_csv):
        generate_dataset_json(input_csv)
    else:
        print(f"❌ 找不到输入文件: {input_csv}，请检查路径。")