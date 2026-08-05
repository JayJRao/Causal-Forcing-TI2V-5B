#!/usr/bin/env python3
"""
快速分析CSV前N行数据（新版本，包含更多评分字段）
统计帧数和各项评分的分布
"""

import pandas as pd
import numpy as np

# 配置
CSV_PATH = "/m2v_intern/m2v_data/data_version/koala36m_250124.csv"
SAMPLE_SIZE = 10000  # 分析前1万行

def print_distribution(series, name, bins=20):
    """打印数值分布统计"""
    print(f"\n{'='*80}")
    print(f"{name} 分布统计")
    print(f"{'='*80}")

    # 基本统计
    print(f"\n基本统计:")
    print(f"  样本数:        {len(series):>12,}")
    print(f"  缺失值:        {series.isna().sum():>12,}")
    print(f"  最小值:        {series.min():>12.4f}")
    print(f"  最大值:        {series.max():>12.4f}")
    print(f"  平均值:        {series.mean():>12.4f}")
    print(f"  中位数:        {series.median():>12.4f}")
    print(f"  标准差:        {series.std():>12.4f}")

    # 百分位数
    print(f"\n百分位数:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = series.quantile(p/100)
        print(f"  P{p:>2}:           {value:>12.4f}")


def main():
    print("=" * 80)
    print(f"数据样本分析 (前 {SAMPLE_SIZE:,} 行) - 新版本")
    print("=" * 80)

    # 读取前N行
    print(f"\n读取CSV文件: {CSV_PATH}")
    print(f"读取行数: {SAMPLE_SIZE:,}")

    try:
        df = pd.read_csv(CSV_PATH, nrows=SAMPLE_SIZE)
        print(f"✓ 读取成功: {len(df):,} 行, {len(df.columns)} 列")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        return

    # 显示列名
    print(f"\n列名: {', '.join(df.columns.tolist())}")

    # 统计基本信息
    print(f"\n{'='*80}")
    print("数据集基本信息")
    print(f"{'='*80}")
    print(f"  总行数:        {len(df):>12,}")
    print(f"  横向视频数:    {(df['width'] > df['height']).sum():>12,} ({(df['width'] > df['height']).sum()/len(df)*100:>5.1f}%)")
    print(f"  纵向视频数:    {(df['width'] < df['height']).sum():>12,} ({(df['width'] < df['height']).sum()/len(df)*100:>5.1f}%)")
    print(f"  方形视频数:    {(df['width'] == df['height']).sum():>12,} ({(df['width'] == df['height']).sum()/len(df)*100:>5.1f}%)")

    # 统计帧数分布
    if 'total_frame_num' in df.columns:
        print_distribution(df['total_frame_num'], '总帧数 (total_frame_num)', bins=20)

        # 常见帧数阈值统计
        print(f"\n常见帧数阈值统计:")
        thresholds = [50, 100, 150, 200, 250, 300, 500, 1000]
        for threshold in thresholds:
            count = (df['total_frame_num'] > threshold).sum()
            pct = count / len(df) * 100
            print(f"  > {threshold:>4} 帧: {count:>8,} ({pct:>5.1f}%)")

    # 统计各项评分分布
    score_fields = [
        ('video_training_suitability_score', '训练适合度评分'),
        ('clarity_score', '清晰度评分'),
        ('aesthetic_score', '美学评分'),
        ('motion_score', '运动评分')
    ]

    for field, name in score_fields:
        if field in df.columns:
            print_distribution(df[field], name, bins=20)

            # 常见阈值统计
            if field == 'video_training_suitability_score':
                thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            elif field == 'clarity_score':
                thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
            elif field == 'aesthetic_score':
                thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
            elif field == 'motion_score':
                thresholds = [5, 10, 15, 20, 25, 30]
            else:
                continue

            print(f"\n常见阈值统计:")
            for threshold in thresholds:
                count = (df[field] >= threshold).sum()
                pct = count / len(df) * 100
                print(f"  >= {threshold:>5}: {count:>8,} ({pct:>5.1f}%)")

    # 组合筛选统计
    print(f"\n{'='*80}")
    print("组合筛选统计示例")
    print(f"{'='*80}")

    if all(field in df.columns for field in ['total_frame_num', 'video_training_suitability_score', 'clarity_score']):
        print("\n横向视频 + 帧数100-150 + 不同评分组合:")
        print(f"{'训练评分':>10} | {'清晰度':>10} | {'美学评分':>10} | {'数据量':>12} | {'占比':>8}")
        print("-" * 60)

        combos = [
            (4.0, 0.8, 4.0),
            (4.0, 0.85, 4.5),
            (4.5, 0.8, 4.0),
            (4.5, 0.85, 4.5),
        ]

        for train_th, clarity_th, aes_th in combos:
            mask = (df['width'] > df['height']) & \
                   (df['total_frame_num'] > 100) & \
                   (df['total_frame_num'] <= 150) & \
                   (df['video_training_suitability_score'] >= train_th) & \
                   (df['clarity_score'] >= clarity_th) & \
                   (df['aesthetic_score'] >= aes_th)
            count = mask.sum()
            pct = count / len(df) * 100
            print(f"  >= {train_th:>4.1f} | >= {clarity_th:>6.2f} | >= {aes_th:>6.1f} | {count:>8,} | {pct:>6.2f}%")

    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
