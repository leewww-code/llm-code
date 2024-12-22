import pandas as pd

# 本地 CSV 文件路径
local_csv_path = "mmlu_ZH-CN.csv"  # 替换为实际的文件路径

# 加载数据集
df = pd.read_csv(local_csv_path)

# 筛选主题为 "moral_scenarios" 的数据
filtered_df = df[df['Subject'] == "moral_scenarios"]

# 选取前 100 行
top_100_moral_scenarios = filtered_df.head(100)

# 保存为新的 CSV 文件
output_file = "moral_scenarios_top100.csv"
top_100_moral_scenarios.to_csv(output_file, index=False, encoding="utf-8")

print(f"筛选完成，已保存到 {output_file}")
