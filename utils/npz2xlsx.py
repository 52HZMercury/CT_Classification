import numpy as np
import pandas as pd
import os


def convert_eval_npz_to_xlsx(npz_path, xlsx_path):
    # 1. 加载 npz 文件
    data = np.load(npz_path, allow_pickle=True)

    # 2. 提取数据并构建字典
    # 确保列名和你要求的一致
    combined_dict = {
        "patient_ids": data["patient_ids"],
        "y_true": data["y_true"],
        "y_probs": data["y_probs"]
    }

    # 3. 转换为 DataFrame
    # 因为我们在推理时是按顺序 extend 进列表的，这里的数据天然就是一一对应的
    df = pd.DataFrame(combined_dict)

    # 4. 可选：增加一列“预测结果” (概率 > 0.5 为 1) 方便你直接看对错
    df["y_pred"] = (df["y_probs"] >= 0.5).astype(int)
    # 增加一列“是否预测正确”
    df["is_correct"] = (df["y_true"] == df["y_pred"])

    # 5. 保存为 Excel
    # index=False 表示不保存左侧的行索引数字
    df.to_excel(xlsx_path, index=False, engine='openpyxl')

    print(f"✅ 转换完成！")
    print(f"📊 样本总数: {len(df)}")
    print(f"准确率 (ACC): {df['is_correct'].mean():.2%}")
    print(f"📂 Excel 已保存至: {xlsx_path}")


if __name__ == "__main__":
    input_npz = "/workdir2/cn24/program/CT_Classification/logs/exp_260328-1752/visualization/eval_data.npz"
    output_xlsx = "/workdir2/cn24/program/CT_Classification/logs/exp_260328-1752/visualization/eval_data.xlsx"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)

    convert_eval_npz_to_xlsx(input_npz, output_xlsx)