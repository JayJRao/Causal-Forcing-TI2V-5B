#!/usr/bin/env python3
"""
删除 checkpoint 目录中数字部分不能被指定值整除的子目录。

默认目录：
logs/ar_diffusion_tf_chunkwise_1.3b_koala36m_335860
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_target = repo_root / "logs/ar_diffusion_tf_chunkwise_1.3b_koala36m_335860"

    parser = argparse.ArgumentParser(
        description="删除 checkpoint_model_数字 里数字不能被指定值整除的目录。"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target,
        help=f"要清理的目录，默认: {default_target}",
    )
    parser.add_argument(
        "--divisor",
        type=int,
        default=1000,
        help="整除阈值，默认: 1000",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行实际删除。不加该参数时只打印将删除的目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.resolve()

    if not target_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")
    if args.divisor <= 0:
        raise ValueError(f"divisor 必须 > 0，当前: {args.divisor}")

    pattern = re.compile(r"^checkpoint_model_(\d+)$")
    to_delete: list[Path] = []

    for item in sorted(target_dir.iterdir()):
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        step = int(match.group(1))
        if step % args.divisor != 0:
            to_delete.append(item)

    if not to_delete:
        print("没有需要删除的目录。")
        return

    mode = "执行删除" if args.execute else "仅预览"
    print(f"[{mode}] 目标目录: {target_dir}")
    print(f"将删除 {len(to_delete)} 个目录：")
    for path in to_delete:
        print(f"- {path}")

    if not args.execute:
        print("\n预览模式，不会删除任何目录。加入 --execute 参数后将执行删除。")
        return

    for path in to_delete:
        shutil.rmtree(path)
    print(f"\n删除完成，共删除 {len(to_delete)} 个目录。")


if __name__ == "__main__":
    main()
