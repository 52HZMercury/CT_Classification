import pandas as pd
import os

# ================= 配置区域 =================
# CSV 文件路径（如果您使用的是 Excel，请看下方的注释代码）
file_path = '/workdir2/cn24/data/30daysSuccess/new_data_30success.xlsx'
# 图像文件夹路径
image_dir = '/workdir2/cn24/data/SCU/image'


# ===========================================

def check_missing_files(file_path, img_folder):
    try:
        # 1. 读取数据
        # 如果是 Excel 文件，请取消下面这行的注释并注释掉 pd.read_csv 那行
        df = pd.read_excel(file_path, sheet_name='data_huaxi_final', dtype={'ID': str})

        # # 读取 CSV，强制将 ID 转为字符串以防止前导零丢失（例如 00123 变成 123）
        # df = pd.read_csv(file_path, dtype={'ID': str})

        # 清理 ID 列：去除空格，并去掉空值
        csv_ids = set(df['ID'].dropna().str.strip().tolist())

        # 2. 检查文件夹是否存在
        if not os.path.exists(img_folder):
            print(f"错误: 找不到目录 {img_folder}")
            return

        # 3. 获取文件夹下的所有 .nii.gz 文件名（去除后缀）
        # 假设文件命名格式为: {ID}.nii.gz
        folder_files = os.listdir(img_folder)
        folder_ids = set()
        for f in folder_files:
            if f.endswith('.nii.gz'):
                # 去掉 .nii.gz 得到 ID 部分
                id_part = f.replace('.nii.gz', '')
                folder_ids.add(id_part)

        # 4. 计算差异：在表格中存在但文件夹中不存在的
        missing_ids = csv_ids - folder_ids

        # 5. 输出统计结果
        print("-" * 30)
        print(f"表格中的 ID 总数: {len(csv_ids)}")
        print(f"文件夹中的 .nii.gz 总数: {len(folder_ids)}")
        print("-" * 30)

        if missing_ids:
            print(f"发现 {len(missing_ids)} 个 ID 缺少对应的图像文件：")
            # 排序输出，方便查看
            for mid in sorted(list(missing_ids)):
                print(f"未找到文件: {mid}.nii.gz")
        else:
            print("恭喜！所有 ID 都有对应的 .nii.gz 文件。")

    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == "__main__":
    check_missing_files(file_path, image_dir)