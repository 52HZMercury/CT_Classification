import pandas as pd
import json
import os
import random
from sklearn.model_selection import train_test_split


def process_dataframe(df, base_data_dir, feature_cols, label_col):
    """
    通用数据处理函数：处理单个 DataFrame 并生成对应的 JSON entry 列表
    """
    # 移除关键信息缺失的行
    df = df.dropna(subset=['ID', label_col])

    # 针对每一列进行处理
    for col in feature_cols:
        # 尝试将“Yes/No/Male/Female”转换为 1/0
        if df[col].dtype == object:
            sample_val = str(df[col].iloc[0]).strip().lower()
            if sample_val in ['yes', 'no', 'male', 'female']:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'male': 1, 'female': 0, 'Yes': 1, 'No': 0})

        # 强制转换为数值型，无法转换的会变成 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # 填充缺失值为 0
        df[col] = df[col].fillna(0)

    # 构造数据列表
    data_list = []
    for _, row in df.iterrows():
        patient_id = str(row['ID']).split('.')[0].strip()

        # 标签映射 (兼容 excel 中写的是 yes 或者是 1 的情况)
        label_str = str(row[label_col]).strip().lower()
        label_val = 1 if label_str in ['yes', '1', '1.0', 'true'] else 0

        # 获取原始数值列表
        tabular_vector = [float(row[c]) for c in feature_cols]

        entry = {
            "image": os.path.join(base_data_dir, "image", f"{patient_id}.nii.gz"),
            "CA": os.path.join(base_data_dir, "CA", f"{patient_id}.nii.gz"),
            "CAC": os.path.join(base_data_dir, "CAC", f"{patient_id}.nii.gz"),
            "tabular_features": tabular_vector,
            "label": label_val
        }
        data_list.append(entry)

    return data_list


def generate_multimodal_json_raw(input_file, output_dir='../metadata', output_filename='new_pvl_bad_valve_performance.json'):
    # 1. 检查并创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 读取两个 Sheet 的
    df_huaxi = pd.read_excel(input_file, sheet_name="data_huaxi_final", dtype={'ID': str})
    df_external = pd.read_excel(input_file, sheet_name="data_external_final", dtype={'ID': str})

    # 3. 定义特征列与标签列
    feature_cols = [
        'Male', 'Age', 'BMI', 'STS_score', 'Hypertension', 'Diabetes', 'COPD', 'Coronary_artery_disease',
        'Chronic_kidney_disease', 'Prior_atrial_fibrillation', 'Peripheral_vascular_disease', 'Prior_stroke_TIA',
        'Aortic_valve_calcification_volume', 'calcified_raphe', 'Annulus_angulation', 'Annular_perimeter',
        'Annular_area', 'SOV_perimeter', 'STJ_diameter', 'Left_coronary_artery_ostium_height',
        'Right_coronary_artery_ostium_height', 'Maximal_diameter_of_ascending_aorta', 'LVOT_perimeter',
        'Aortic_regurgitation_moderate', 'Mean_aortic_valve_gradient', 'Peak_aortic_valve_velocity',
        'LVEF', 'LVEDD', 'IVS'
    ]
    label_col = 'Bad_valve_performance'

    # 获取在两个表中都存在的共同特征列，防止某个表缺列报错
    final_feature_cols = []
    for col in feature_cols:
        if col in df_huaxi.columns and col in df_external.columns:
            final_feature_cols.append(col)
        else:
            print(f"⚠️ 跳过缺失列: {col}")

    # 4. 处理 data_huaxi_final (分一部分训练，一部分验证)
    huaxi_data_list = process_dataframe(
        df=df_huaxi,
        base_data_dir="/workdir2/cn24/data/SCU",
        feature_cols=final_feature_cols,
        label_col=label_col
    )

    # 划分 huaxi 数据
    pos_samples_huaxi = [d for d in huaxi_data_list if d['label'] == 1]
    neg_samples_huaxi = [d for d in huaxi_data_list if d['label'] == 0]

    # 分别对正负样本进行 8:2 划分
    train_pos_hauxi, val_pos_huaxi = train_test_split(pos_samples_huaxi, test_size=0.2, random_state=42)
    train_neg_huaxi, val_neg_huaxi = train_test_split(neg_samples_huaxi, test_size=0.2, random_state=42)


    # 5. 处理 data_external_final (全部作为验证集)
    other_data_list = process_dataframe(
        df=df_external,
        base_data_dir="/workdir2/cn24/data/30daysSuccess",
        feature_cols=final_feature_cols,
        label_col=label_col
    )

    # 划分 other 数据
    pos_samples_other = [d for d in other_data_list if d['label'] == 1]
    neg_samples_other = [d for d in other_data_list if d['label'] == 0]

    # 分别对正负样本进行 8:2 划分
    train_pos_other, val_pos_other = train_test_split(pos_samples_other, test_size=0.2, random_state=42)
    train_neg_other, val_neg_other = train_test_split(neg_samples_other, test_size=0.2, random_state=42)


    # 6. 合并验证集并打乱顺序
    train_data = train_pos_hauxi + train_neg_huaxi + train_pos_other + train_neg_other
    val_data = val_pos_huaxi + val_neg_huaxi + val_pos_other + val_neg_other

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
    print(f"📊 总样本数: {len(huaxi_data_list) + len(other_data_list)}")
    print(f"📊 训练集: {len(train_data)} (阳性: {len(train_pos_hauxi) + len(train_pos_other)} | 阴性: {len(train_neg_huaxi) + len(train_neg_other)})")
    print(f"📊 验证集: {len(val_data)} (阳性: {len(val_pos_huaxi) + len(val_pos_other)} | 阴性: {len(val_neg_huaxi) + len(val_neg_other)})")
    print("-" * 30)


if __name__ == "__main__":
    # 更新了你指定的文件名
    input_xlsx = '/workdir2/cn24/data/30daysSuccess/new_data_30success.xlsx'
    if os.path.exists(input_xlsx):
        generate_multimodal_json_raw(input_xlsx)
    else:
        print(f"❌ 找不到文件: {input_xlsx}")