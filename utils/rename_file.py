import os
import pandas as pd


def rename_nii_files():
    # 1. 路径与文件配置
    target_dir = "/workdir2/cn24/data/30daysSuccess/image"
    # 请根据您的实际情况填写包含对应关系的csv文件路径 (例如您上传的 external final csv)
    excel_path = "/workdir2/cn24/data/30daysSuccess/data_30success.xlsx"

    # 2. 读取表格
    # 指定 dtype={'ID': str} 防止像 "08006" 这样的ID被 pandas 自动读取为数字 "8006"
    try:
        df = pd.read_excel(excel_path, sheet_name="data_external_final", dtype={'ID': str})
    except Exception as e:
        print(f"读取Excel文件失败，请检查文件路径是否正确: {e}")
        return

    # 检查是否存在需要的列
    if '文件名称' not in df.columns or 'ID' not in df.columns:
        print("错误: 表格中找不到 '文件名称' 或 'ID' 列！")
        return

    # 过滤掉空值
    df = df.dropna(subset=['文件名称', 'ID'])

    # 3. 建立映射字典: 文件名称 -> ID
    # 使用 strip() 去除可能存在的前后空格
    name_to_id = dict(zip(df['文件名称'].astype(str).str.strip(),
                          df['ID'].astype(str).str.strip()))

    # 4. 遍历文件夹并重命名
    if not os.path.exists(target_dir):
        print(f"错误: 目标文件夹 {target_dir} 不存在！")
        return

    rename_count = 0
    skip_count = 0

    for filename in os.listdir(target_dir):
        if filename.endswith(".nii.gz"):
            # 获取去除后缀的纯名字 (如 'Ai_Ren_Ying.nii.gz' -> 'Ai_Ren_Ying')
            base_name = filename.replace(".nii.gz", "").strip()

            # 查找映射字典
            if base_name in name_to_id:
                new_id = name_to_id[base_name]
                new_filename = f"{new_id}.nii.gz"

                old_path = os.path.join(target_dir, filename)
                new_path = os.path.join(target_dir, new_filename)

                # 如果新名字已经存在，可能需要避免覆盖
                if os.path.exists(new_path) and old_path != new_path:
                    print(f"警告: 目标文件 {new_filename} 已存在，跳过 {filename} 的重命名")
                    continue

                # 执行重命名
                os.rename(old_path, new_path)
                print(f"成功: {filename} -> {new_filename}")
                rename_count += 1
            else:
                print(f"跳过: 未在表格中找到 {filename} 对应的ID")
                skip_count += 1

    print("-" * 30)
    print(f"执行完毕！共重命名 {rename_count} 个文件，跳过 {skip_count} 个文件。")


if __name__ == "__main__":
    rename_nii_files()