import os

# 需要统计的文件夹路径
folder_path = "/root/seed"  # 例如 "C:/projects/apis"

# 用于存储 API 名称的集合（去重）
api_names = set()

# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        # 假设文件名格式为 api名_seedn.py
        parts = filename.split("_seed")
        if len(parts) == 2:
            api_name = parts[0]
            api_names.add(api_name)

# 输出统计信息
output_file = os.path.join(folder_path, "api-result.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"总共有 {len(api_names)} 个不同的 API:\n")
    for api in sorted(api_names):
        f.write(api + "\n")

print(f"统计完成，结果已写入 {output_file}")