"""
将 JSONL 文件中的 event 文字渲染到视频左上角，生成新视频。

用法：
    python scripts/render_events_on_video.py \
        --jsonl /m2v_intern/xuyifan09/projects/zjm_video_seg/data/20251129_480p_14s-30s-100w-video_metadata_full_15s_landscape_2500.jsonl \
        --output_dir /m2v_intern/xuyifan09/projects/zjm_video_seg/data/20251129_480p_14s-30s-100w-video_metadata_full_15s_landscape_2500 \
        [--num_workers 4] \
        [--limit 10]           # 仅处理前 N 条（调试用）
        [--entry_idx 0]        # 仅处理第 N 条（调试用）
        [--event_count single] # 只处理恰好有 1 个 event 的视频
        [--event_count multi]  # 只处理有 2+ 个 event 的视频
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# 核心：用 FFmpeg drawtext 滤镜渲染 event 字幕
# ──────────────────────────────────────────────────────────────

def assign_rows(events: list[dict]) -> list[int]:
    """
    每个 event 独占一行，行号一旦分配不再复用。
    events 已按 start_time 排序。
    返回与 events 等长的行号列表（0-based）。
    """
    return list(range(len(events)))


def build_drawtext_filter(events: list[dict], font_size: int = 32) -> str:
    """
    将多个 event 转成 FFmpeg drawtext 滤镜链。
    同一时刻有多个 event 时，各自渲染到不同行。
    长文本自动折行（每行最多 48 个字符）。
    """
    # 按 start_time 排序后分配行号
    sorted_events = sorted(events, key=lambda e: float(e["start_time"]))
    row_indices = assign_rows(sorted_events)

    # 每行高度：font_size + line_spacing(6) + boxborderw*2(16) + 行间距(8)
    row_height = font_size + 30

    filters = []
    for idx, (evt, row) in enumerate(zip(sorted_events, row_indices)):
        text = evt["event"]
        start = float(evt["start_time"])
        end   = float(evt["end_time"])

        # 时间前缀，格式：#1 [0.0s - 8.9s] event text
        time_prefix = f"#{idx + 1} [{start:.1f}s - {end:.1f}s] "
        text = time_prefix + text

        lines = textwrap.wrap(text, width=48)
        escaped = r"\n".join(
            line.replace("\\", r"\\")
                .replace("'", "\u2019")   # ' → ' 避免 FFmpeg 单引号解析问题
                .replace(":", r"\:")
                .replace("%", "%%")
            for line in lines
        )

        y = 20 + row * row_height

        f = (
            f"drawtext=fontsize={font_size}"
            f":fontcolor=white"
            f":box=1:boxcolor=black@0.55:boxborderw=8"
            f":x=20:y={y}"
            f":line_spacing=6"
            f":text='{escaped}'"
            f":enable='between(t,{start},{end})'"
        )
        filters.append(f)

    return ",".join(filters)


def process_entry(entry: dict, output_dir: str, font_size: int = 32) -> tuple[bool, str]:
    """
    处理单条 JSONL 记录，生成带字幕的新视频。
    返回 (success, message)。
    """
    video_path = entry["video_path"]
    events     = entry.get("result", [])

    if not os.path.isfile(video_path):
        return False, f"视频不存在: {video_path}"

    if not events:
        return False, f"无 event，跳过: {video_path}"

    video_name = Path(video_path).name
    out_path   = os.path.join(output_dir, video_name)

    # 已存在则跳过
    if os.path.isfile(out_path):
        return True, f"已存在，跳过: {out_path}"

    vf = build_drawtext_filter(events, font_size=font_size)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        out_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-500:]
            return False, f"FFmpeg 失败 ({video_name}): {err}"
        return True, f"完成: {out_path}"
    except subprocess.TimeoutExpired:
        return False, f"超时: {video_path}"
    except Exception as e:
        return False, f"异常 ({video_path}): {e}"


# ──────────────────────────────────────────────────────────────
# 过滤辅助
# ──────────────────────────────────────────────────────────────

def _has_overlap(events: list[dict]) -> bool:
    """判断 events 中是否存在任意两个时间段重叠。"""
    sorted_evts = sorted(events, key=lambda e: float(e["start_time"]))
    for i in range(len(sorted_evts) - 1):
        if float(sorted_evts[i]["end_time"]) > float(sorted_evts[i + 1]["start_time"]):
            return True
    return False


def _match_event_count(entry: dict, mode: str) -> bool:
    events = entry.get("result", [])
    if mode == "single":
        return len(events) == 1
    if mode == "multi_no_overlap":
        return len(events) >= 2 and not _has_overlap(events)
    if mode == "multi_overlap":
        return len(events) >= 2 and _has_overlap(events)
    return True


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="将 event 渲染到视频左上角")
    parser.add_argument("--jsonl",       required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_dir",  required=True, help="输出视频目录")
    parser.add_argument("--num_workers", type=int, default=4,  help="并行进程数")
    parser.add_argument("--font_size",   type=int, default=32, help="字体大小（像素）")
    parser.add_argument("--limit",       type=int, default=None, help="最多处理前 N 条（调试用）")
    parser.add_argument("--entry_idx",   type=int, default=None, help="只处理第 N 条（调试用，0-based）")
    parser.add_argument(
        "--event_count",
        choices=["single", "multi_no_overlap", "multi_overlap"],
        default=None,
        help=(
            "过滤 event 类型：\n"
            "  single           = 恰好 1 个 event\n"
            "  multi_no_overlap = 2+ 个 event 且互不重叠\n"
            "  multi_overlap    = 2+ 个 event 且存在时间重叠"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取 JSONL
    entries = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # 调试模式：只处理指定条目
    if args.entry_idx is not None:
        entries = [entries[args.entry_idx]]
    elif args.limit is not None:
        entries = entries[: args.limit]

    # 按 event 类型过滤
    if args.event_count is not None:
        before = len(entries)
        entries = [e for e in entries if _match_event_count(e, args.event_count)]
        print(f"过滤 {args.event_count}：{before} -> {len(entries)} 条")

    total = len(entries)
    print(f"共 {total} 条记录，输出目录: {args.output_dir}，并行进程: {args.num_workers}")

    success = fail = 0

    if args.num_workers <= 1:
        for entry in entries:
            ok, msg = process_entry(entry, args.output_dir, args.font_size)
            print(("[OK] " if ok else "[FAIL] ") + msg)
            if ok:
                success += 1
            else:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_entry, e, args.output_dir, args.font_size): e
                for e in entries
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                ok, msg = fut.result()
                print(f"[{done}/{total}] " + ("[OK] " if ok else "[FAIL] ") + msg)
                if ok:
                    success += 1
                else:
                    fail += 1

    print(f"\n完成：成功 {success}，失败 {fail}，共 {total}")


if __name__ == "__main__":
    main()
