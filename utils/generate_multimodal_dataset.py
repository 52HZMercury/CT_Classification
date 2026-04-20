import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split


def generate_multimodal_json_raw(input_file, output_dir='../metadata', output_filename='other_pvl_30daysSuccess.json'):
    # 1. 检查并创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 加载数据 (强制 DicomID 为字符串)
    # df = pd.read_excel(input_file, sheet_name="data_huaxi_final", dtype={'ID': str})
    df = pd.read_excel(input_file, sheet_name="data_external_final", dtype={'ID': str})


    feature_cols = [
        'Male', 'Age', 'BMI', 'STS_score', 'Hypertension', 'Diabetes', 'COPD', 'Coronary_artery_disease',
        'Chronic_kidney_disease', 'Prior_atrial_fibrillation', 'Peripheral_vascular_disease', 'Prior_stroke_TIA',
        'Aortic_valve_calcification_volume', 'calcified_raphe', 'Annulus_angulation', 'Annular_perimeter',
        'Annular_area', 'SOV_perimeter', 'STJ_diameter', 'Left_coronary_artery_ostium_height',
        'Right_coronary_artery_ostium_height', 'Maximal_diameter_of_ascending_aorta', 'LVOT_perimeter',
        'Aortic_regurgitation_moderate', 'Mean_aortic_valve_gradient', 'Peak_aortic_valve_velocity',
        'LVEF', 'LVEDD', 'IVS'
    ]

    label_col = 'Device_success_at_30_days'

    # 4. 数据预处理
    # 移除关键信息缺失的行
    df = df.dropna(subset=['ID', label_col])

    # 针对每一列进行处理
    final_feature_cols = []
    for col in feature_cols:
        if col in df.columns:
            # 尝试将“Yes/No”转换为 1/0 (医学表格常见情况)
            if df[col].dtype == object:
                # 统一转小写并去除空格
                sample_val = str(df[col].iloc[0]).strip().lower()
                if sample_val in ['yes', 'no', 'male', 'female']:
                    df[col] = df[col].map({'yes': 1, 'no': 0, 'male': 1, 'female': 0, 'Yes': 1, 'No': 0})

            # 强制转换为数值型，无法转换的(如纯文本)会变成 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # 填充缺失值：原始数据中如果有空值，必须填充一个数值（如0或中位数），否则神经网络无法计算
            # 这里建议用 0 填充，代表“无”或“未知”
            df[col] = df[col].fillna(0)
            final_feature_cols.append(col)
        else:
            print(f"⚠️ 跳过缺失列: {col}")

    # 【关键修改】：删除了之前的 StandardScaler 步骤，直接使用 df[final_feature_cols]

    # 5. 构造数据列表
    data_list = []
    base_data_dir = "/workdir2/cn24/data/30daysSuccess"

    for _, row in df.iterrows():
        patient_id = str(row['ID']).split('.')[0].strip()

        # 标签映射
        label_val = 1 if str(row[label_col]).strip().lower() == 'yes' else 0

        # 获取原始数值列表
        tabular_vector = [float(row[c]) for c in final_feature_cols]

        entry = {
            "image": os.path.join(base_data_dir, "image", f"{patient_id}.nii.gz"),
            # "pred": os.path.join(base_data_dir, "label", f"{patient_id}_pred.nii.gz"),
            "CA": os.path.join(base_data_dir, "CA", f"{patient_id}.nii.gz"),
            "CAC": os.path.join(base_data_dir, "CAC", f"{patient_id}.nii.gz"),
            "tabular_features": tabular_vector,  # 这里的 x 是原始数值
            "label": label_val
        }
        data_list.append(entry)

    # 6. 划分数据集 (确保阳性样本的 80% 进入训练集，且整体比例 8:2)
    # 将正负样本分开
    pos_samples = [d for d in data_list if d['label'] == 1]
    neg_samples = [d for d in data_list if d['label'] == 0]

    # 只要175例
    # neg_samples = neg_samples[:175]
    # pos_samples = pos_samples[:208]

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

    # 7. 构建最终 JSON 结构
    final_output = {
        "features_list": final_feature_cols,
        "training": train_data,
        "validation": val_data
    }

    # 8. 写入并打印统计信息
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 多模态数据集已生成！保存至: {output_path}")
    print(f"📊 总样本数: {len(data_list)}")
    print(f"   - 阳性(1): {len(pos_samples)} | 阴性(0): {len(neg_samples)}")
    print(f"📊 训练集: {len(train_data)} (包含 {len(train_pos)} 个阳性)")
    print(f"📊 验证集: {len(val_data)} (包含 {len(val_pos)} 个阳性)")
    print("-" * 30)


if __name__ == "__main__":
    input_xlsx = '/workdir2/cn24/data/SCU/data_30success.xlsx'
    if os.path.exists(input_xlsx):
        generate_multimodal_json_raw(input_xlsx)