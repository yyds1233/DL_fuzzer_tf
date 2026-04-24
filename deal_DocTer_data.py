#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from pathlib import Path

# 固定前缀
FIXED_PREFIX = r"/home/smoke_workdir/tensorflow/violate_constr/"

# 匹配：
# open('.../something_workdir/xxx.p'
# open(".../something_workdir/xxx.p"
#
# 其中：
# - /home/smoke_workdir/tensorflow/violate_constr/ 是固定的
# - 中间的 xxx_workdir/ 可以变化
# - 只保留最后的 .p 文件名
pattern = re.compile(
    rf"""open\(\s*       # open(
        (['"])           # 捕获引号
        {re.escape(FIXED_PREFIX)}
        [^'"]*?_workdir/ # 可变目录，例如 tf.cast.yaml_workdir/
        ([^/'"]+\.p)     # 捕获最终的 .p 文件名
        \1               # 与前面相同的引号
    """,
    re.VERBOSE
)

def replace_in_file(file_path: Path) -> int:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="latin-1")
        except Exception as e:
            print(f"[跳过] 无法读取文件: {file_path}，原因: {e}")
            return 0
    except Exception as e:
        print(f"[跳过] 无法读取文件: {file_path}，原因: {e}")
        return 0

    def repl(match: re.Match) -> str:
        quote = match.group(1)
        p_file = match.group(2)
        return f"open({quote}{p_file}{quote}"

    new_content, count = pattern.subn(repl, content)

    if count > 0 and new_content != content:
        try:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"[修改] {file_path} -> 替换 {count} 处")
        except Exception as e:
            print(f"[失败] 写入文件失败: {file_path}，原因: {e}")
            return 0

    return count

def process_directory(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        print(f"目录不存在: {root}")
        return
    if not root.is_dir():
        print(f"传入路径不是目录: {root}")
        return

    total_files = 0
    total_replacements = 0

    for py_file in root.rglob("*.py"):
        total_files += 1
        total_replacements += replace_in_file(py_file)

    print("\n处理完成")
    print(f"扫描 .py 文件数: {total_files}")
    print(f"总替换次数: {total_replacements}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <目标目录>")
        sys.exit(1)

    process_directory(sys.argv[1])