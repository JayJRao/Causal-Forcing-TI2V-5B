#!/usr/bin/env python3
"""
CSV 数据集融合脚本
将多个 CSV 文件按指定比例融合，输出文件名自动由各文件末尾数值和比例拼接生成。

用法示例:
  # 两个文件按 1:1 融合（自动确定最大可用量）
  python scripts/merge_csvs.py --files datasets/a.csv datasets/b.csv --ratios 1 1

  # 三个文件按 2:1:1 融合，指定总量为 10 万条
  python scripts/merge_csvs.py --files datasets/a.csv datasets/b.csv datasets/c.csv --ratios 2 1 1 --total 100000

  # 自定义输出路径
  python scripts/merge_csvs.py --files datasets/a.csv datasets/b.csv --ratios 3 1 --output datasets/my_merged.csv
"""

import pandas as pd
import numpy as np
import argparse
import re
import os
from pathlib import Path
from datetime import datetime


def extract_tail_number(filepath):
    """从文件名中提取末尾的数字，如 _335860.csv -> 335860"""
    stem = Path(filepath).stem
    match = re.search(r'_(\d+)$', stem)
    if match:
        return match.group(1)
    return stem  # fallback: 用整个文件名stem


def generate_output_name(files, ratios, output_dir, total_count, seed):
    """生成输出文件名：merged_{时间}_{总数}_{种子}_{num1}x{ratio1}_{num2}x{ratio2}.csv"""
    parts = []
    for f, r in zip(files, ratios):
        num = extract_tail_number(f)
        # 比例展示：整数去掉小数点，否则保留
        r_str = str(int(r)) if r == int(r) else str(r)
        parts.append(f"{num}x{r_str}")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"merged_{timestamp}_{total_count}_{seed}_" + "_".join(parts) + f".csv"
    return os.path.join(output_dir, filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CSV 数据集融合工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--files', nargs='+', required=True,
        help='输入 CSV 文件路径列表（需与 --ratios 一一对应）'
    )
    parser.add_argument(
        '--ratios', nargs='+', type=float, required=True,
        help='各 CSV 文件对应的融合比例（相对值，如 1 2 1 表示 1:2:1）'
    )
    parser.add_argument(
        '--total', type=int, default=None,
        help='期望输出的总行数（不指定则自动取各文件在比例下的最大可用量）'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出文件路径（不指定则自动生成至 datasets/ 目录）'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子，保证可复现 [默认: 42]'
    )
    parser.add_argument(
        '--no-shuffle', action='store_true',
        help='不打乱输出结果的顺序'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.files) != len(args.ratios):
        print(f"错误：--files 数量 ({len(args.files)}) 与 --ratios 数量 ({len(args.ratios)}) 不匹配")
        return

    print("=" * 70)
    print("CSV 数据集融合工具")
    print("=" * 70)

    # 归一化比例
    total_ratio = sum(args.ratios)
    norm_ratios = [r / total_ratio for r in args.ratios]

    # 读取各文件行数（只读 header，快速）
    print("\n[1/4] 读取各文件信息...")
    file_sizes = []
    for f in args.files:
        size = sum(1 for _ in open(f)) - 1  # 减去 header
        file_sizes.append(size)
        num = extract_tail_number(f)
        print(f"  {Path(f).name}")
        print(f"    实际行数: {size:,}  |  比例权重: {args.ratios[args.files.index(f)]}")

    # 确定各文件采样数量
    print("\n[2/4] 计算采样数量...")

    # 以瓶颈文件为上限，确保比例严格成立且不超采任何文件
    # 瓶颈 = min(file_size / norm_ratio)，即哪个文件最"撑不住"这个比例
    max_possible = min(size / r for size, r in zip(file_sizes, norm_ratios))
    if args.total is not None:
        # 用户指定了 total，但仍不能超过文件容量上限
        max_possible = min(max_possible, args.total)

    target_counts = [round(max_possible * r) for r in norm_ratios]
    target_counts[-1] = sum(target_counts) - sum(target_counts[:-1])  # 修正尾项
    target_total = sum(target_counts)

    if args.total is not None and target_total < args.total:
        print(f"  ⚠ 指定 total={args.total:,}，但受文件容量限制，实际最多可采 {target_total:,} 条")

    print(f"\n  目标总量: {target_total:,} 条")
    for f, size, count, ratio in zip(args.files, file_sizes, target_counts, args.ratios):
        pct = count / target_total * 100
        flag = " ← 瓶颈文件（全部使用）" if count >= size else ""
        print(f"  {Path(f).name}")
        print(f"    目标: {count:,} 条 ({pct:.1f}%)  |  文件共: {size:,} 条{flag}")

    # 采样并合并
    print("\n[3/4] 采样并合并...")
    np.random.seed(args.seed)
    dfs = []
    actual_counts = []

    for f, count, size in zip(args.files, target_counts, file_sizes):
        df = pd.read_csv(f)
        if count >= size:
            sampled = df
        else:
            sampled = df.sample(n=count, random_state=args.seed)
        dfs.append(sampled)
        actual_counts.append(len(sampled))
        print(f"  ✓ {Path(f).name}: 采样 {len(sampled):,} 条")

    merged = pd.concat(dfs, ignore_index=True)

    if not args.no_shuffle:
        merged = merged.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"  ✓ 已随机打乱顺序")

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(args.files[0]) or "datasets"
        output_path = generate_output_name(args.files, args.ratios, output_dir, len(merged), args.seed)

    # 保存
    print(f"\n[4/4] 保存结果...")
    merged.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"融合完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总行数:   {len(merged):,} 条")
    print(f"\n  各来源占比:")
    for f, count in zip(args.files, actual_counts):
        print(f"    {Path(f).name}: {count:,} 条 ({count/len(merged)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
