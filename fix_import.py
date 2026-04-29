import argparse
import re
from pathlib import Path


REQUIRED_IMPORTS = [
    {
        "name": "tensorflow",
        "line": "import tensorflow as tf\n",
        "patterns": [
            r"^\s*import\s+tensorflow\s+as\s+tf\s*$",
        ],
    },
    {
        "name": "numpy",
        "line": "import numpy as np\n",
        "patterns": [
            r"^\s*import\s+numpy\s+as\s+np\s*$",
        ],
    },
    {
        "name": "sys",
        "line": "import sys\n",
        "patterns": [
            r"^\s*import\s+sys\s*$",
        ],
    },
    {
        "name": "math",
        "line": "import math\n",
        "patterns": [
            r"^\s*import\s+math\s*$",
        ],
    },
]


def has_import(content: str, patterns) -> bool:
    return any(re.search(p, content, flags=re.MULTILINE) for p in patterns)


def find_insert_pos(lines):
    """
    把新增 import 放到已有 import 区域后面。
    如果没有 import，就放到文件头注释之后。
    """
    last_import_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i

    if last_import_idx is not None:
        return last_import_idx + 1

    # 没有 import，则跳过文件头注释和空行
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("#"):
            i += 1
            continue
        break

    return i


def fix_file(path: Path, dry_run: bool = False):
    original = path.read_text(encoding="utf-8")
    content = original

    imports_to_add = []

    for item in REQUIRED_IMPORTS:
        if not has_import(content, item["patterns"]):
            imports_to_add.append(item["line"])

    if not imports_to_add:
        return {
            "file": str(path),
            "changed": False,
            "added": [],
        }

    lines = content.splitlines(keepends=True)
    insert_pos = find_insert_pos(lines)

    # 插入 import，并尽量保持格式清楚
    block = imports_to_add

    if insert_pos < len(lines) and lines[insert_pos].strip() != "":
        block = imports_to_add + ["\n"]

    lines[insert_pos:insert_pos] = block
    new_content = "".join(lines)

    if not dry_run:
        path.write_text(new_content, encoding="utf-8")

    return {
        "file": str(path),
        "changed": True,
        "added": [x.strip() for x in imports_to_add],
    }


def main():
    parser = argparse.ArgumentParser(
        description="批量给 py 文件补充 import tensorflow as tf / import sys / import math"
    )

    parser.add_argument(
        "input_dir",
        help="需要处理的 py 文件夹",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印要修改的文件，不实际写入",
    )

    args = parser.parse_args()

    root = Path(args.input_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"不是有效文件夹：{root}")

    py_files = sorted(root.rglob("*.py") if args.recursive else root.glob("*.py"))

    print(f"发现 {len(py_files)} 个 py 文件")

    changed_count = 0

    for path in py_files:
        try:
            result = fix_file(path, dry_run=args.dry_run)
        except Exception as e:
            print(f"[失败] {path}: {e}")
            continue

        if result["changed"]:
            changed_count += 1
            print(f"[修改] {result['file']} added={result['added']}")

    print()
    print(f"完成，修改文件数：{changed_count}")

    if args.dry_run:
        print("当前是 dry-run，没有实际写入文件")


if __name__ == "__main__":
    main()
