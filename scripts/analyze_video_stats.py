#!/usr/bin/env python3
"""
分析视频时长、帧数、分辨率的分布，输出一张包含多子图的统计图。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

CSV_PATH = "/m2v_intern_v3/zhangjiaming09/Video-Causal-Dataset/20251129_480p_14s-30s-100w/video_metadata_full_15s_landscape-video_stats.csv"
OUTPUT_IMG = os.path.join(os.path.dirname(CSV_PATH), "0312_10w_1080p_stats.png")

# ── 读取数据 ────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df["error"].isna() | (df["error"] == "")]   # 过滤出错行
df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
df["nb_frames"]  = pd.to_numeric(df["nb_frames"],  errors="coerce")
df["width"]      = pd.to_numeric(df["width"],       errors="coerce")
df["height"]     = pd.to_numeric(df["height"],      errors="coerce")
df = df.dropna(subset=["duration_s", "nb_frames"])

total = len(df)
print(f"有效样本: {total:,} 条")
print(f"时长(s)  — min={df['duration_s'].min():.1f}  max={df['duration_s'].max():.1f}"
      f"  mean={df['duration_s'].mean():.1f}  median={df['duration_s'].median():.1f}")
print(f"帧数     — min={df['nb_frames'].min():.0f}  max={df['nb_frames'].max():.0f}"
      f"  mean={df['nb_frames'].mean():.1f}  median={df['nb_frames'].median():.1f}")

# 分辨率组合
df["resolution"] = df["width"].astype(int).astype(str) + "x" + df["height"].astype(int).astype(str)
res_counts = df["resolution"].value_counts().sort_values(ascending=False)
print(f"分辨率种类: {len(res_counts)}")
print(res_counts.head(10).to_string())

# ── 画布布局：2行3列，右侧一列分辨率占满整列 ────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle(f"KlingVA 0312_10w_1080p  —  {total:,} videos", fontsize=14, fontweight="bold")
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35, width_ratios=[2, 2, 2.5])
axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
ax_res = fig.add_subplot(gs[:, 2])
BIN_COLOR = "#4C72B0"
CDF_COLOR = "#DD8452"

def add_stats_text(ax, series):
    txt = (f"mean={series.mean():.2f}\n"
           f"median={series.median():.2f}\n"
           f"std={series.std():.2f}\n"
           f"min={series.min():.2f}\n"
           f"max={series.max():.2f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# ── (0,0) 时长直方图 ────────────────────────────────────────
ax = axes[0][0]
ax.hist(df["duration_s"], bins=60, color=BIN_COLOR, edgecolor="white", linewidth=0.3)
ax.set_title("Duration Distribution")
ax.set_xlabel("Duration (s)")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
add_stats_text(ax, df["duration_s"])

# ── (0,1) 时长 CDF ─────────────────────────────────────────
ax = axes[0][1]
sorted_dur = np.sort(df["duration_s"])
cdf = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
ax.plot(sorted_dur, cdf * 100, color=CDF_COLOR, linewidth=1.5)
ax.set_title("Duration CDF")
ax.set_xlabel("Duration (s)")
ax.set_ylabel("Cumulative (%)")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, linestyle="--", alpha=0.4)
# 标注百分位线
for p, ls in [(25, ":"), (50, "--"), (75, ":")]:
    val = np.percentile(df["duration_s"], p)
    ax.axvline(val, color="gray", linestyle=ls, linewidth=0.8)
    ax.text(val, p, f" p{p}={val:.1f}s", fontsize=7, color="gray", va="bottom")

# ── (1,0) 帧数直方图 ────────────────────────────────────────
ax = axes[1][0]
ax.hist(df["nb_frames"], bins=60, color=BIN_COLOR, edgecolor="white", linewidth=0.3)
ax.set_title("Frame Count Distribution")
ax.set_xlabel("Frames")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
add_stats_text(ax, df["nb_frames"])

# ── (1,1) 帧数 CDF ─────────────────────────────────────────
ax = axes[1][1]
sorted_frm = np.sort(df["nb_frames"])
cdf = np.arange(1, len(sorted_frm) + 1) / len(sorted_frm)
ax.plot(sorted_frm, cdf * 100, color=CDF_COLOR, linewidth=1.5)
ax.set_title("Frame Count CDF")
ax.set_xlabel("Frames")
ax.set_ylabel("Cumulative (%)")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, linestyle="--", alpha=0.4)
for p, ls in [(25, ":"), (50, "--"), (75, ":")]:
    val = np.percentile(df["nb_frames"], p)
    ax.axvline(val, color="gray", linestyle=ls, linewidth=0.8)
    ax.text(val, p, f" p{p}={val:.0f}", fontsize=7, color="gray", va="bottom")

# ── (:,2) 分辨率水平柱状图（占满右侧整列）──────────────────
top_n = res_counts.head(20)
y_pos = range(len(top_n))
bars = ax_res.barh(list(y_pos), top_n.values, color=BIN_COLOR, edgecolor="white", linewidth=0.3)
ax_res.set_yticks(list(y_pos))
ax_res.set_yticklabels(top_n.index, fontsize=8)
ax_res.invert_yaxis()  # 最多的在顶部
ax_res.set_title(f"Resolution Distribution\n(Top {len(top_n)} of {len(res_counts)})")
ax_res.set_xlabel("Count")
ax_res.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
# 右侧标注占比
for bar, cnt in zip(bars, top_n.values):
    pct = cnt / total * 100
    ax_res.text(bar.get_width() + total * 0.001, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", ha="left", va="center", fontsize=7)

plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
print(f"图片已保存: {OUTPUT_IMG}")
