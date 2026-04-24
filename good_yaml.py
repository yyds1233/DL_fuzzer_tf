import os
import re
from pathlib import Path

def find_high_confidence_yaml(folder_path, threshold=0.9):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"文件夹不存在或不是目录: {folder_path}")
        return [], 0

    # 匹配 confidence: 0.95 这种格式
    pattern = re.compile(r'^\s*confidence\s*:\s*([0-9]*\.?[0-9]+)\s*$', re.IGNORECASE)

    matched_files = []

    for yaml_file in folder.glob("*.yaml"):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        confidence = float(match.group(1))
                        if confidence > threshold:
                            matched_files.append(yaml_file.name)
                        break  # 找到 confidence 后就不用继续读了
        except Exception as e:
            print(f"读取文件失败 {yaml_file}: {e}")

    for yml_file in folder.glob("*.yml"):
        try:
            with open(yml_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        confidence = float(match.group(1))
                        if confidence > threshold:
                            matched_files.append(yml_file.name)
                        break
        except Exception as e:
            print(f"读取文件失败 {yml_file}: {e}")

    return matched_files, len(matched_files)


if __name__ == "__main__":
    folder_path = "/root/yaml_list_4/08_final"  # 改成你的文件夹路径
    files, count = find_high_confidence_yaml(folder_path, threshold=0.9)

    print("confidence YAML 文件有：")
    for name in files:
        print(name)

    print(f"\n总数量: {count}")