import os

# 指定要处理的文件夹路径
folder_path = "/root/seed"  # 例如 "C:/projects/apis"

# 要添加的导入语句
imports_to_add = ["import tensorflow as tf", "import numpy as np"]

# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 检查导入语句是否已存在
        new_lines = []
        for imp in imports_to_add:
            if not any(line.strip() == imp for line in lines):
                new_lines.append(imp + "\n")

        # 如果有需要添加的导入语句，则写回文件
        if new_lines:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines + lines)

print("所有 .py 文件已更新导入语句。")
