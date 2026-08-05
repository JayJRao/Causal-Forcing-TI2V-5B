#!/usr/bin/env python3
"""
快速分析CSV前N行数据
统计帧数和训练适合度评分的分布
"""

import pandas as pd
import numpy as np

# 配置
CSV_PATH = "/m2v_intern/m2v_data/data_version/koala36m.csv"
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
    print(f"  最小值:        {series.min():>12.2f}")
    print(f"  最大值:        {series.max():>12.2f}")
    print(f"  平均值:        {series.mean():>12.2f}")
    print(f"  中位数:        {series.median():>12.2f}")
    print(f"  标准差:        {series.std():>12.2f}")

    # 百分位数
    print(f"\n百分位数:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = series.quantile(p/100)
        print(f"  P{p:>2}:           {value:>12.2f}")

    # 区间分布
    print(f"\n区间分布:")
    hist, bin_edges = np.histogram(series.dropna(), bins=bins)
    total = len(series.dropna())

    for i in range(len(hist)):
        left = bin_edges[i]
        right = bin_edges[i+1]
        count = hist[i]
        pct = count / total * 100
        bar_length = int(pct / 2)  # 每2%一个字符
        bar = '█' * bar_length

        if i == len(hist) - 1:
            # 最后一个区间包含右边界
            print(f"  [{left:>10.1f}, {right:>10.1f}]: {count:>8,} ({pct:>5.1f}%) {bar}")
        else:
            print(f"  [{left:>10.1f}, {right:>10.1f}): {count:>8,} ({pct:>5.1f}%) {bar}")


def main():
    print("=" * 80)
    print(f"数据样本分析 (前 {SAMPLE_SIZE:,} 行)")
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

    # 统计分辨率分布
    print(f"\n分辨率分布 (Top 10):")
    resolution_counts = df.groupby(['width', 'height']).size().sort_values(ascending=False).head(10)
    for (w, h), count in resolution_counts.items():
        orientation = "横向" if w > h else ("纵向" if w < h else "方形")
        print(f"  {w:>4}x{h:<4} ({orientation}): {count:>6,} ({count/len(df)*100:>5.1f}%)")

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

    # 统计训练适合度评分分布
    if 'video_training_suitability_score' in df.columns:
        print_distribution(df['video_training_suitability_score'],
                          '训练适合度评分 (video_training_suitability_score)',
                          bins=20)

        # 常见评分阈值统计
        print(f"\n常见评分阈值统计:")
        thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        for threshold in thresholds:
            count = (df['video_training_suitability_score'] >= threshold).sum()
            pct = count / len(df) * 100
            print(f"  >= {threshold:.1f}: {count:>8,} ({pct:>5.1f}%)")

    # 统计时长分布
    if 'duration' in df.columns:
        duration_sec = df['duration'] / 1000  # 转换为秒
        print_distribution(duration_sec, '视频时长 (秒)', bins=20)

    # 组合筛选统计
    print(f"\n{'='*80}")
    print("组合筛选统计")
    print(f"{'='*80}")

    if 'total_frame_num' in df.columns and 'video_training_suitability_score' in df.columns:
        print("\n横向视频 + 不同帧数阈值 + 不同评分阈值的数据量:")
        print(f"{'帧数阈值':>10} | {'评分阈值':>10} | {'数据量':>12} | {'占比':>8}")
        print("-" * 50)

        frame_thresholds = [50, 100, 150, 200]
        score_thresholds = [2.5, 3.0, 3.5, 4.0]

        for frame_th in frame_thresholds:
            for score_th in score_thresholds:
                mask = (df['width'] > df['height']) & \
                       (df['total_frame_num'] > frame_th) & \
                       (df['video_training_suitability_score'] >= score_th)
                count = mask.sum()
                pct = count / len(df) * 100
                print(f"  > {frame_th:>3} 帧 | >= {score_th:>4.1f}  | {count:>8,} | {pct:>6.1f}%")

    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
