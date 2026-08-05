#!/usr/bin/env python3
"""
将 all_dimension_extended 下的 mp4 文件按 txt 行内容重命名，
并复制到 all_dimension_extended-rename 目录下。

设计：每个 all_dimension_extended 文件夹均有相同数量的 mp4（与 txt 行数一致），
      txt 按行顺序与 mp4 按文件名排序后一一对应，逐文件夹独立处理。

用法:
  python rename_by_txt.py <txt_path> [--videos-root <path>] [--dry-run]

参数:
  txt_path        txt 文件路径，行数须与每个文件夹的 mp4 数量一致
  --videos-root   videos 根目录，默认为脚本同级的 videos 文件夹
  --dry-run       只打印操作，不实际复制/创建目录
"""

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def find_mp4_folders(videos_root: Path) -> List[Path]:
    """找出所有包含 all_dimension_extended 且含 mp4 的叶子文件夹，按路径排序。
    使用 os.walk(followlinks=True) 以支持软链接目录。
    """
    seen = set()
    result = []
    all_mp4s = []
    for dirpath, _dirs, files in os.walk(str(videos_root), followlinks=True):
        folder = Path(dirpath)
        if "all_dimension_extended" in folder.parts:
            mp4s = [folder / f for f in files if f.endswith(".mp4")]
            if mp4s and folder not in seen:
                seen.add(folder)
                all_mp4s.append((folder, mp4s))
    # 按文件夹路径排序
    all_mp4s.sort(key=lambda x: str(x[0]))
    return [folder for folder, _ in all_mp4s]


def sanitize_name(raw: str) -> str:
    """用整行内容作文件名，替换文件名非法字符为 _，去掉首尾空白。"""
    name = raw.strip()
    # 替换 / \ : * ? " < > | 及制表符为下划线
    name = re.sub(r'[/\\:*?"<>|\t]', "_", name)
    return name


def dest_folder(src_folder: Path, videos_root: Path) -> Path:
    """将源路径中的 all_dimension_extended 替换为 all_dimension_extended-rename。"""
    rel = str(src_folder.relative_to(videos_root))
    new_rel = rel.replace("all_dimension_extended", "all_dimension_extended-rename", 1)
    return videos_root / new_rel


def main():
    parser = argparse.ArgumentParser(description="按 txt 行内容重命名 mp4 文件")
    parser.add_argument(
        "txt_path", nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "prompts" / "vbench" / "all_dimension.txt")
    )
    parser.add_argument(
        "--videos-root",
        default=str(Path(__file__).resolve().parent.parent / "videos"),
        help="videos 根目录（默认: <repo>/videos）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印，不执行")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件夹（0=全部）")
    args = parser.parse_args()

    txt_path = Path(args.txt_path)
    videos_root = Path(args.videos_root)

    # 读取 txt
    if not txt_path.exists():
        print(f"[ERROR] txt 文件不存在: {txt_path}", file=sys.stderr)
        sys.exit(1)
    lines = [l for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_lines = len(lines)
    print(f"[INFO] txt 有效行数: {n_lines}  ({txt_path})")

    # 找所有文件夹
    print(f"[INFO] 扫描目录: {videos_root}")
    folders = find_mp4_folders(videos_root)
    if not folders:
        print("[ERROR] 未找到任何 all_dimension_extended 文件夹", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 找到 {len(folders)} 个文件夹")

    RED = "\033[31m"
    RESET = "\033[0m"

    # 校验每个文件夹的 mp4 数量，数量不一致时警告并跳过
    mismatch_folders = set()
    for folder in folders:
        mp4s = sorted(folder.glob("*.mp4"))
        if len(mp4s) != n_lines:
            print(
                f"{RED}[WARN] 行数不一致，已跳过: {folder}\n"
                f"       mp4 数量 {len(mp4s)} ≠ txt 行数 {n_lines}{RESET}",
                file=sys.stderr,
            )
            mismatch_folders.add(folder)
    folders = [f for f in folders if f not in mismatch_folders]

    # 生成新文件名：始终以 -{n} 结尾，重复行从 0 开始递增，唯一行固定 -0
    base_names = [sanitize_name(l) for l in lines]
    name_count: Dict[str, int] = {}
    for name in base_names:
        name_count[name] = name_count.get(name, 0) + 1

    name_seen: Dict[str, int] = {}
    new_names: List[str] = []
    for name in base_names:
        idx = name_seen.get(name, 0)
        name_seen[name] = idx + 1
        if name_count[name] > 1:
            print(f"[WARN] 重复行，加后缀: {name}-{idx}.mp4")
        new_names.append(f"{name}-{idx}.mp4")

    # 逐文件夹处理
    if args.limit > 0:
        folders = folders[: args.limit]
    total_folders = len(folders)
    skipped = 0
    warned = 0
    for fi, folder in enumerate(folders, 1):
        mp4s = sorted(folder.glob("*.mp4"))
        dst_dir = dest_folder(folder, videos_root)
        print(f"\n[{fi}/{total_folders}] {folder.name}")
        print(f"      -> {dst_dir}")

        # 目标目录已存在时：检查数量后决定跳过还是报警
        if dst_dir.exists():
            existing = len(list(dst_dir.glob("*.mp4")))
            if existing == n_lines:
                print(f"  [SKIP] 目标已存在且文件数匹配（{existing} 个），跳过")
                skipped += 1
                continue
            else:
                print(f"  [WARN] 目标已存在但文件数不符（期望 {n_lines}，实际 {existing}），将重新处理")
                warned += 1

        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=n_lines, desc=folder.name, unit="file", leave=False) as pbar:
            for mp4, new_name in zip(mp4s, new_names):
                dst = dst_dir / new_name
                if not args.dry_run:
                    shutil.copy2(src=mp4, dst=dst)
                pbar.update(1)
        print(f"[{fi}/{total_folders}] 完成: {folder.name}  ->  {dst_dir}")

    processed = total_folders - skipped
    if args.dry_run:
        print(f"\n[DRY-RUN] 共 {total_folders} 个文件夹，未实际操作。")
    else:
        print(f"\n[DONE] 共 {total_folders} 个文件夹：处理 {processed} 个，跳过 {skipped} 个，警告 {warned} 个。")
    if mismatch_folders:
        print(
            f"{RED}[WARN] 共 {len(mismatch_folders)} 个文件夹因 mp4 数量与 txt 行数不一致被跳过:{RESET}",
            file=sys.stderr,
        )
        for f in sorted(mismatch_folders):
            print(f"{RED}       {f}{RESET}", file=sys.stderr)


if __name__ == "__main__":
    main()
