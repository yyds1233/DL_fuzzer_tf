import argparse
import os
import re
from pathlib import Path


def remove_atheris_instrument_imports(content: str) -> str:
    """
    把：

    with atheris.instrument_imports():
        import tensorflow as tf
        import xxx

    改成：

    import tensorflow as tf
    import xxx

    只处理 with 块下面连续缩进的 import / from import 语句。
    """

    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    changed = False

    pattern = re.compile(r"^(\s*)with\s+atheris\.instrument_imports\(\)\s*:\s*(#.*)?$")

    while i < len(lines):
        line = lines[i]
        match = pattern.match(line.rstrip("\n"))

        if not match:
            new_lines.append(line)
            i += 1
            continue

        base_indent = match.group(1)
        i += 1

        block_lines = []

        while i < len(lines):
            next_line = lines[i]

            # 空行保留在块中
            if next_line.strip() == "":
                block_lines.append(next_line)
                i += 1
                continue

            # with 块里的内容至少要比 with 多一层缩进
            if not next_line.startswith(base_indent + "    "):
                break

            stripped = next_line[len(base_indent + "    "):]

            # 只处理 import / from import 行
            if stripped.lstrip().startswith("import ") or stripped.lstrip().startswith("from "):
                block_lines.append(base_indent + stripped)
                i += 1
                changed = True
                continue

            # 如果 with 块里出现非 import 内容，为了安全，保留原始 with 结构
            new_lines.append(line)
            new_lines.extend(block_lines)
            break
        else:
            # 文件结束
            pass

        # 正常情况下，把取消缩进后的 import 行写回
        if block_lines:
            # 如果刚才因为非 import 内容中断并已经写回了原始 with，这里避免重复写
            if new_lines and new_lines[-1:] == block_lines[-1:]:
                continue
            new_lines.extend(block_lines)
        else:
            # with 下面没有内容，直接删除这个 with
            changed = True

    return "".join(new_lines), changed


def ensure_testoneinput_decorator(content: str) -> tuple[str, bool]:
    """
    确保 def TestOneInput(...) 前面有 @atheris.instrument_func。
    如果已经有，不重复添加。
    """

    lines = content.splitlines(keepends=True)
    new_lines = []
    changed = False

    def_pattern = re.compile(r"^(\s*)def\s+TestOneInput\s*\(")

    for idx, line in enumerate(lines):
        match = def_pattern.match(line)

        if not match:
            new_lines.append(line)
            continue

        indent = match.group(1)

        # 找到 def 前面最近的非空行
        j = len(new_lines) - 1
        while j >= 0 and new_lines[j].strip() == "":
            j -= 1

        already_has_decorator = (
            j >= 0 and new_lines[j].strip() == "@atheris.instrument_func"
        )

        if not already_has_decorator:
            new_lines.append(f"{indent}@atheris.instrument_func\n")
            changed = True

        new_lines.append(line)

    return "".join(new_lines), changed


def fix_one_file(py_path: Path, dry_run: bool = False) -> dict:
    original = py_path.read_text(encoding="utf-8")

    content, changed_imports = remove_atheris_instrument_imports(original)
    content, changed_decorator = ensure_testoneinput_decorator(content)

    changed = content != original

    if changed and not dry_run:
        py_path.write_text(content, encoding="utf-8")

    return {
        "file": str(py_path),
        "changed": changed,
        "changed_imports": changed_imports,
        "changed_decorator": changed_decorator,
    }


def collect_py_files(input_dir: str, recursive: bool = True):
    root = Path(input_dir)

    if recursive:
        return sorted(root.rglob("*.py"))

    return sorted(root.glob("*.py"))


def main():
    parser = argparse.ArgumentParser(
        description="批量修复 harness 脚本：去掉 atheris.instrument_imports，并补充 @atheris.instrument_func"
    )

    parser.add_argument(
        "input_dir",
        help="需要处理的 py 脚本文件夹",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="只处理当前目录，不递归子目录",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印会修改哪些文件，不实际写入",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"不是有效文件夹：{input_dir}")

    py_files = collect_py_files(
        str(input_dir),
        recursive=not args.no_recursive,
    )

    print(f"发现 {len(py_files)} 个 .py 文件")

    changed_count = 0
    import_changed_count = 0
    decorator_changed_count = 0

    for py_file in py_files:
        try:
            result = fix_one_file(py_file, dry_run=args.dry_run)
        except Exception as e:
            print(f"[失败] {py_file}: {e}")
            continue

        if result["changed"]:
            changed_count += 1

            if result["changed_imports"]:
                import_changed_count += 1

            if result["changed_decorator"]:
                decorator_changed_count += 1

            print(
                f"[修改] {result['file']} "
                f"imports={result['changed_imports']} "
                f"decorator={result['changed_decorator']}"
            )

    print()
    print("完成")
    print(f"总 py 文件数：{len(py_files)}")
    print(f"修改文件数：{changed_count}")
    print(f"去掉 instrument_imports 文件数：{import_changed_count}")
    print(f"补充 decorator 文件数：{decorator_changed_count}")

    if args.dry_run:
        print("当前是 dry-run，没有实际写入文件")


if __name__ == "__main__":
    main()
