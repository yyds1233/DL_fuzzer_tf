import os
import json
import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError(
        "缺少 PyYAML，请先安装：pip install pyyaml"
    )


def api_name_to_txt_path(api_name: str, txt_root: str) -> str:
    """
    把 api_name 转成 txt 路径：
    tf.argsort -> /root/tf_api_txt/tf_argsort.txt
    """
    txt_name = api_name.replace(".", "_") + ".txt"
    return os.path.join(txt_root, txt_name)


def collect_yaml_api_info(input_dirs, output_json, txt_root="/root/tf_api_txt"):
    results = []

    for input_dir in input_dirs:
        input_dir = os.path.abspath(input_dir)

        if not os.path.isdir(input_dir):
            print(f"[跳过] 文件夹不存在：{input_dir}")
            continue

        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                    continue

                yaml_path = os.path.join(root, filename)

                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as e:
                    print(f"[读取失败] {yaml_path}: {e}")
                    continue

                if not isinstance(data, dict):
                    print(f"[跳过] YAML 顶层不是 dict：{yaml_path}")
                    continue

                api_name = data.get("api_name")

                if not api_name:
                    print(f"[跳过] 没有 api_name 字段：{yaml_path}")
                    continue

                item = {
                    "api": api_name,
                    "yaml": yaml_path,
                    "txt": api_name_to_txt_path(api_name, txt_root),
                }

                results.append(item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成，共收集 {len(results)} 条记录")
    print(f"输出文件：{os.path.abspath(output_json)}")


def main():
    parser = argparse.ArgumentParser(
        description="遍历文件夹下所有 YAML 文件，提取 api_name 并生成 JSON"
    )

    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="一个或多个输入文件夹"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="api_yaml_txt_mapping.json",
        help="输出 JSON 文件路径，默认 api_yaml_txt_mapping.json"
    )

    parser.add_argument(
        "--txt-root",
        default="/root/tf_api_txt",
        help="txt 文件所在根目录，默认 /root/tf_api_txt"
    )

    args = parser.parse_args()

    collect_yaml_api_info(
        input_dirs=args.input_dirs,
        output_json=args.output,
        txt_root=args.txt_root,
    )


if __name__ == "__main__":
    main()