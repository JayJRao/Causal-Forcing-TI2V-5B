#!/usr/bin/env python3
"""
视频数据筛选脚本（增强版）
支持多维度筛选条件：
- 视频方向（horizontal/vertical/all）
- 帧数范围
- 训练适合度评分（可选，支持单向或范围）
- 清晰度评分（可选，支持单向或范围）
- 美学评分（可选，支持单向或范围）
- 运动评分（可选，支持单向或范围）
- 可选的视频文件存在性检查
"""

import pandas as pd
import os
import subprocess
import json
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path

# 默认配置
CSV_PATH = "/m2v_intern/m2v_data/data_version/koala36m_250124.csv"
NUM_WORKERS = min(32, cpu_count())  # 使用CPU核心数，最多32个worker
CHUNK_SIZE = 10000  # 每次读取10000行


def check_video_exists(video_path):
    """
    只检查视频文件是否存在（快速版本）
    返回: (video_path, is_valid, error_msg)
    """
    if os.path.exists(video_path):
        return (video_path, True, "OK")
    else:
        return (video_path, False, "File not found")


def check_video_batch(video_paths):
    """批量检查视频（用于多进程）"""
    results = []
    for path in video_paths:
        results.append(check_video_exists(path))
    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='视频数据筛选工具（增强版 - 支持多评分筛选）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 基础筛选：横向视频，100-150帧，训练分数>4.0
  python filter_videos.py --orientation horizontal --frames 100 150 --train-min 4.0

  # 高质量高动态：训练>4.0，清晰度>0.5，运动>30
  python filter_videos.py --frames 100 150 --train-min 4.0 --clarity-min 0.5 --motion-min 30

  # 范围筛选：美学分数在4.0-5.0之间
  python filter_videos.py --frames 100 150 --aesthetic-min 4.0 --aesthetic-max 5.0

  # 组合筛选并保存
  python filter_videos.py --frames 100 150 --train-min 4.0 --clarity-min 0.5 --motion-min 30 --save
        '''
    )

    # 基础筛选参数
    parser.add_argument(
        '--orientation',
        choices=['horizontal', 'vertical', 'all'],
        default='horizontal',
        help='视频方向: horizontal(横向), vertical(纵向), all(所有) [默认: horizontal]'
    )

    parser.add_argument(
        '--frames-min',
        type=int,
        default=100,
        help='最小帧数 [默认: 100]'
    )

    parser.add_argument(
        '--frames-max',
        type=int,
        default=150,
        help='最大帧数 [默认: 150]'
    )

    # 训练适合度评分
    parser.add_argument(
        '--train-min',
        type=float,
        default=None,
        help='训练适合度评分最小值 (不设置则不限制)'
    )

    parser.add_argument(
        '--train-max',
        type=float,
        default=None,
        help='训练适合度评分最大值 (不设置则不限制)'
    )

    # 清晰度评分
    parser.add_argument(
        '--clarity-min',
        type=float,
        default=None,
        help='清晰度评分最小值 (不设置则不限制)'
    )

    parser.add_argument(
        '--clarity-max',
        type=float,
        default=None,
        help='清晰度评分最大值 (不设置则不限制)'
    )

    # 美学评分
    parser.add_argument(
        '--aesthetic-min',
        type=float,
        default=None,
        help='美学评分最小值 (不设置则不限制)'
    )

    parser.add_argument(
        '--aesthetic-max',
        type=float,
        default=None,
        help='美学评分最大值 (不设置则不限制)'
    )

    # 运动评分
    parser.add_argument(
        '--motion-min',
        type=float,
        default=None,
        help='运动评分最小值 (不设置则不限制)'
    )

    parser.add_argument(
        '--motion-max',
        type=float,
        default=None,
        help='运动评分最大值 (不设置则不限制)'
    )

    # 其他选项
    parser.add_argument(
        '--check-files',
        action='store_true',
        help='检查视频文件是否存在（会增加处理时间）'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='保存筛选结果到CSV文件'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 [默认: 自动生成包含筛选参数的文件名]'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=CSV_PATH,
        help=f'输入CSV文件路径 [默认: {CSV_PATH}]'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=NUM_WORKERS,
        help=f'并行处理的进程数 [默认: {NUM_WORKERS}]'
    )

    return parser.parse_args()


def generate_filename(args, result_count):
    """根据筛选条件生成文件名（不包含特殊符号和中文）"""
    parts = []

    # 方向
    orientation_short = {
        'horizontal': 'h',
        'vertical': 'v',
        'all': 'all'
    }[args.orientation]
    parts.append(orientation_short)

    # 帧数
    parts.append(f"{args.frames_min}-{args.frames_max}f")

    # 训练分数
    if args.train_min is not None or args.train_max is not None:
        if args.train_min is not None and args.train_max is not None:
            parts.append(f"train{args.train_min}-{args.train_max}")
        elif args.train_min is not None:
            parts.append(f"train{args.train_min}")
        else:
            parts.append(f"trainmax{args.train_max}")

    # 清晰度
    if args.clarity_min is not None or args.clarity_max is not None:
        if args.clarity_min is not None and args.clarity_max is not None:
            parts.append(f"clarity{args.clarity_min}-{args.clarity_max}")
        elif args.clarity_min is not None:
            parts.append(f"clarity{args.clarity_min}")
        else:
            parts.append(f"claritymax{args.clarity_max}")

    # 美学分数
    if args.aesthetic_min is not None or args.aesthetic_max is not None:
        if args.aesthetic_min is not None and args.aesthetic_max is not None:
            parts.append(f"aes{args.aesthetic_min}-{args.aesthetic_max}")
        elif args.aesthetic_min is not None:
            parts.append(f"aes{args.aesthetic_min}")
        else:
            parts.append(f"aesmax{args.aesthetic_max}")

    # 运动分数
    if args.motion_min is not None or args.motion_max is not None:
        if args.motion_min is not None and args.motion_max is not None:
            parts.append(f"motion{args.motion_min}-{args.motion_max}")
        elif args.motion_min is not None:
            parts.append(f"motion{args.motion_min}")
        else:
            parts.append(f"motionmax{args.motion_max}")

    # 数据量
    parts.append(f"{result_count}")

    filename = "koala36m_filtered_" + "_".join(parts) + ".csv"
    return filename


def main():
    args = parse_args()

    # 记录是否使用自动生成的文件名
    auto_filename = (args.output is None)

    print("=" * 80)
    print("视频数据筛选工具 (增强版 - 分块读取 + 多评分筛选)")
    print("=" * 80)

    # 显示筛选条件
    print(f"\n筛选条件:")
    print(f"  视频方向:      {args.orientation}")
    print(f"  帧数范围:      {args.frames_min} - {args.frames_max} 帧")

    # 训练分数
    if args.train_min is not None or args.train_max is not None:
        train_desc = []
        if args.train_min is not None:
            train_desc.append(f">= {args.train_min}")
        if args.train_max is not None:
            train_desc.append(f"<= {args.train_max}")
        print(f"  训练分数:      {' 且 '.join(train_desc)}")
    else:
        print(f"  训练分数:      不限制")

    # 清晰度
    if args.clarity_min is not None or args.clarity_max is not None:
        clarity_desc = []
        if args.clarity_min is not None:
            clarity_desc.append(f">= {args.clarity_min}")
        if args.clarity_max is not None:
            clarity_desc.append(f"<= {args.clarity_max}")
        print(f"  清晰度:        {' 且 '.join(clarity_desc)}")
    else:
        print(f"  清晰度:        不限制")

    # 美学分数
    if args.aesthetic_min is not None or args.aesthetic_max is not None:
        aes_desc = []
        if args.aesthetic_min is not None:
            aes_desc.append(f">= {args.aesthetic_min}")
        if args.aesthetic_max is not None:
            aes_desc.append(f"<= {args.aesthetic_max}")
        print(f"  美学分数:      {' 且 '.join(aes_desc)}")
    else:
        print(f"  美学分数:      不限制")

    # 运动分数
    if args.motion_min is not None or args.motion_max is not None:
        motion_desc = []
        if args.motion_min is not None:
            motion_desc.append(f">= {args.motion_min}")
        if args.motion_max is not None:
            motion_desc.append(f"<= {args.motion_max}")
        print(f"  运动分数:      {' 且 '.join(motion_desc)}")
    else:
        print(f"  运动分数:      不限制")

    print(f"  检查文件:      {'是' if args.check_files else '否'}")
    print(f"  保存结果:      {'是' if args.save else '否'}")

    # 1. 分块读取CSV并筛选
    print(f"\n[1/5] 分块读取CSV文件: {args.input}")
    print(f"  chunk大小: {CHUNK_SIZE:,} 行")

    # 只读取需要的列，节省内存
    columns_needed = ['id', 'height', 'width', 'total_frame_num', 'duration',
                     'video_path', 'video_training_suitability_score',
                     'clarity_score', 'aesthetic_score', 'motion_score', 'caption']

    # 指定数据类型，加快读取速度
    dtype_dict = {
        'id': 'int64',
        'height': 'int32',
        'width': 'int32',
        'total_frame_num': 'int32',
        'duration': 'int32',
        'video_path': 'string',
        'video_training_suitability_score': 'float32',
        'clarity_score': 'float32',
        'aesthetic_score': 'float32',
        'motion_score': 'float32',
        'caption': 'string'
    }

    try:
        # 分块读取并筛选
        chunks_filtered = []
        initial_count = 0
        step_counts = {
            'orientation': 0,
            'frames': 0,
            'train': 0,
            'clarity': 0,
            'aesthetic': 0,
            'motion': 0
        }

        print("\n[2/5] 应用筛选条件 (边读边筛选)...")

        # 创建CSV读取器
        csv_reader = pd.read_csv(
            args.input,
            usecols=columns_needed,
            dtype=dtype_dict,
            chunksize=CHUNK_SIZE
        )

        # 分块处理
        for chunk_idx, chunk in enumerate(tqdm(csv_reader, desc="  读取进度")):
            initial_count += len(chunk)

            # 筛选视频方向
            if args.orientation == 'horizontal':
                chunk_filtered = chunk[chunk['width'] > chunk['height']].copy()
            elif args.orientation == 'vertical':
                chunk_filtered = chunk[chunk['width'] < chunk['height']].copy()
            else:  # all
                chunk_filtered = chunk.copy()
            step_counts['orientation'] += len(chunk_filtered)

            # 筛选帧数范围
            chunk_filtered = chunk_filtered[
                (chunk_filtered['total_frame_num'] > args.frames_min) &
                (chunk_filtered['total_frame_num'] <= args.frames_max)
            ].copy()
            step_counts['frames'] += len(chunk_filtered)

            # 筛选训练分数
            if args.train_min is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['video_training_suitability_score'] >= args.train_min
                ].copy()
            if args.train_max is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['video_training_suitability_score'] <= args.train_max
                ].copy()
            step_counts['train'] += len(chunk_filtered)

            # 筛选清晰度
            if args.clarity_min is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['clarity_score'] >= args.clarity_min
                ].copy()
            if args.clarity_max is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['clarity_score'] <= args.clarity_max
                ].copy()
            step_counts['clarity'] += len(chunk_filtered)

            # 筛选美学分数
            if args.aesthetic_min is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['aesthetic_score'] >= args.aesthetic_min
                ].copy()
            if args.aesthetic_max is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['aesthetic_score'] <= args.aesthetic_max
                ].copy()
            step_counts['aesthetic'] += len(chunk_filtered)

            # 筛选运动分数
            if args.motion_min is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['motion_score'] >= args.motion_min
                ].copy()
            if args.motion_max is not None:
                chunk_filtered = chunk_filtered[
                    chunk_filtered['motion_score'] <= args.motion_max
                ].copy()
            step_counts['motion'] += len(chunk_filtered)

            if len(chunk_filtered) > 0:
                chunks_filtered.append(chunk_filtered)

            # 每处理100个chunk显示一次进度
            if (chunk_idx + 1) % 100 == 0:
                print(f"    已处理: {initial_count:,} 行, 当前符合条件: {step_counts['motion']:,} 行")

        print(f"\n  ✓ 总数据量: {initial_count:,} 条")

        orientation_desc = {
            'horizontal': '横向视频 (width > height)',
            'vertical': '纵向视频 (width < height)',
            'all': '所有视频'
        }[args.orientation]

        print(f"  ✓ {orientation_desc}: {step_counts['orientation']:,} 条 ({step_counts['orientation']/initial_count*100:.1f}%)")
        print(f"  ✓ 帧数范围 ({args.frames_min}-{args.frames_max}): {step_counts['frames']:,} 条 ({step_counts['frames']/initial_count*100:.1f}%)")

        if args.train_min is not None or args.train_max is not None:
            print(f"  ✓ 训练分数筛选后: {step_counts['train']:,} 条 ({step_counts['train']/initial_count*100:.1f}%)")
        if args.clarity_min is not None or args.clarity_max is not None:
            print(f"  ✓ 清晰度筛选后: {step_counts['clarity']:,} 条 ({step_counts['clarity']/initial_count*100:.1f}%)")
        if args.aesthetic_min is not None or args.aesthetic_max is not None:
            print(f"  ✓ 美学分数筛选后: {step_counts['aesthetic']:,} 条 ({step_counts['aesthetic']/initial_count*100:.1f}%)")
        if args.motion_min is not None or args.motion_max is not None:
            print(f"  ✓ 运动分数筛选后: {step_counts['motion']:,} 条 ({step_counts['motion']/initial_count*100:.1f}%)")

        if not chunks_filtered:
            print("\n没有符合条件的数据！")
            return

        # 合并所有筛选后的chunk
        print("\n  合并筛选结果...")
        df_filtered = pd.concat(chunks_filtered, ignore_index=True)
        print(f"  ✓ 合并完成: {len(df_filtered):,} 条")

    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 检查视频文件存在性（可选）
    if args.check_files:
        print(f"\n[3/5] 检查视频文件是否存在 (使用 {args.workers} 个进程)...")
        video_paths = df_filtered['video_path'].tolist()

        # 分批处理
        batch_size = max(1, len(video_paths) // args.workers)
        batches = [video_paths[i:i+batch_size] for i in range(0, len(video_paths), batch_size)]

        # 使用多进程检查
        valid_videos = set()
        invalid_videos = {}

        with Pool(args.workers) as pool:
            results = []
            for batch_result in tqdm(
                pool.imap(check_video_batch, batches),
                total=len(batches),
                desc="  检查进度"
            ):
                results.extend(batch_result)

        # 统计结果
        for video_path, is_valid, error_msg in results:
            if is_valid:
                valid_videos.add(video_path)
            else:
                invalid_videos[video_path] = error_msg

        # 4. 最终筛选
        print("\n[4/5] 生成最终结果...")
        df_final = df_filtered[df_filtered['video_path'].isin(valid_videos)].copy()
    else:
        print("\n[3/5] 跳过文件检查")
        print("\n[4/5] 生成最终结果...")
        df_final = df_filtered
        valid_videos = set(df_final['video_path'].tolist())
        invalid_videos = {}

    # 5. 统计信息
    print("\n[5/5] 统计信息")
    print("=" * 80)
    print(f"\n原始数据总量:           {initial_count:>10,} 条")
    print(f"{orientation_desc}:     {step_counts['orientation']:>10,} 条  ({step_counts['orientation']/initial_count*100:>5.1f}%)")
    print(f"帧数范围筛选:           {step_counts['frames']:>10,} 条  ({step_counts['frames']/initial_count*100:>5.1f}%)")

    if args.train_min is not None or args.train_max is not None:
        print(f"训练分数筛选:           {step_counts['train']:>10,} 条  ({step_counts['train']/initial_count*100:>5.1f}%)")
    if args.clarity_min is not None or args.clarity_max is not None:
        print(f"清晰度筛选:             {step_counts['clarity']:>10,} 条  ({step_counts['clarity']/initial_count*100:>5.1f}%)")
    if args.aesthetic_min is not None or args.aesthetic_max is not None:
        print(f"美学分数筛选:           {step_counts['aesthetic']:>10,} 条  ({step_counts['aesthetic']/initial_count*100:>5.1f}%)")
    if args.motion_min is not None or args.motion_max is not None:
        print(f"运动分数筛选:           {step_counts['motion']:>10,} 条  ({step_counts['motion']/initial_count*100:>5.1f}%)")

    if args.check_files:
        print(f"视频文件有效:           {len(valid_videos):>10,} 条  ({len(valid_videos)/step_counts['motion']*100:>5.1f}% if step_counts['motion'] > 0 else 0)")
    print(f"\n最终筛选结果:           {len(df_final):>10,} 条  ({len(df_final)/initial_count*100:>5.1f}%)")

    # 失败原因统计
    if invalid_videos:
        print(f"\n无效视频数量:           {len(invalid_videos):>10,} 条")
        print("\n失败原因分布:")
        error_stats = {}
        for error_msg in invalid_videos.values():
            error_type = error_msg.split(':')[0]
            error_stats[error_type] = error_stats.get(error_type, 0) + 1

        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type:<30} {count:>8,} 条  ({count/len(invalid_videos)*100:>5.1f}%)")

    # 各项评分分布
    if len(df_final) > 0:
        print(f"\n最终结果的评分分布:")

        # 训练分数
        if 'video_training_suitability_score' in df_final.columns:
            scores = df_final['video_training_suitability_score']
            print(f"\n  训练分数:")
            print(f"    平均: {scores.mean():.2f} | 中位数: {scores.median():.2f} | 范围: {scores.min():.2f}-{scores.max():.2f}")

        # 清晰度
        if 'clarity_score' in df_final.columns:
            scores = df_final['clarity_score']
            print(f"  清晰度:")
            print(f"    平均: {scores.mean():.2f} | 中位数: {scores.median():.2f} | 范围: {scores.min():.2f}-{scores.max():.2f}")

        # 美学分数
        if 'aesthetic_score' in df_final.columns:
            scores = df_final['aesthetic_score']
            print(f"  美学分数:")
            print(f"    平均: {scores.mean():.2f} | 中位数: {scores.median():.2f} | 范围: {scores.min():.2f}-{scores.max():.2f}")

        # 运动分数
        if 'motion_score' in df_final.columns:
            scores = df_final['motion_score']
            print(f"  运动分数:")
            print(f"    平均: {scores.mean():.2f} | 中位数: {scores.median():.2f} | 范围: {scores.min():.2f}-{scores.max():.2f}")

        print("\n视频分辨率分布 (Top 10):")
        resolution_counts = df_final.groupby(['width', 'height']).size().sort_values(ascending=False).head(10)
        for (w, h), count in resolution_counts.items():
            print(f"  - {w}x{h:<20} {count:>8,} 条  ({count/len(df_final)*100:>5.1f}%)")

        print("\n视频时长统计:")
        durations = df_final['duration'] / 1000  # 转换为秒
        print(f"  - 平均时长:             {durations.mean():>10.1f} 秒")
        print(f"  - 中位数时长:           {durations.median():>10.1f} 秒")
        print(f"  - 最短时长:             {durations.min():>10.1f} 秒")
        print(f"  - 最长时长:             {durations.max():>10.1f} 秒")

    print("\n" + "=" * 80)

    # 保存结果（如果需要）
    if args.save and len(df_final) > 0:
        # 如果使用自动文件名，添加数量信息
        if auto_filename:
            args.output = generate_filename(args, len(df_final))

        print(f"\n保存筛选结果到: {args.output}")
        try:
            df_final.to_csv(args.output, index=False)
            print(f"✓ 成功保存 {len(df_final):,} 条数据到: {args.output}")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    elif args.save:
        print(f"\n没有数据可保存")
    else:
        print(f"\n未保存结果（使用 --save 参数保存）")

    return df_final


if __name__ == "__main__":
    df_result = main()
