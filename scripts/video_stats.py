#!/usr/bin/env python3
"""
统计指定目录下所有 MP4 视频的信息并写入 CSV。
信息包括：文件路径、文件名、文件大小(MB)、时长(s)、分辨率、帧率、帧数
"""

import os
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

VIDEO_DIR = "/m2v_intern_v3/zhangjiaming09/Video-Causal-Dataset/20251129_480p_14s-30s-100w/video_metadata_full_15s_landscape"
OUTPUT_CSV = os.path.join(os.path.dirname(VIDEO_DIR), "video_stats.csv")
NUM_WORKERS = 32


def probe_video(filepath):
    """使用 ffprobe 获取视频元信息，返回一行数据 dict。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        "-count_packets",   # 实际统计每个流的包数，比元数据 nb_frames 可靠
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
    except Exception as e:
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_mb": round(os.path.getsize(filepath) / 1024 / 1024, 3),
            "error": str(e),
        }

    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    # 帧率（avg_frame_rate 形如 "30000/1001"）
    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = round(int(num) / int(den), 3) if int(den) != 0 else 0
    except Exception:
        fps = 0

    size_bytes = int(fmt.get("size", os.path.getsize(filepath)))

    # 帧数：用 nb_read_packets（-count_packets 实际统计），比元数据 nb_frames 可靠
    nb_frames_raw = video_stream.get("nb_read_packets", "")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        nb_frames = int(nb_frames_raw)
    else:
        # 降级：尝试元数据 nb_frames
        nb_frames_raw = video_stream.get("nb_frames", "")
        nb_frames = int(nb_frames_raw) if nb_frames_raw and str(nb_frames_raw).isdigit() else ""

    # 时长：由帧数和 fps 反推，不信任容器元数据（容器 duration 可能是原片时长）
    if nb_frames and fps > 0:
        duration = round(nb_frames / fps, 3)
    else:
        duration = round(float(fmt.get("duration", 0)), 3)

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "size_mb": round(size_bytes / 1024 / 1024, 3),
        "duration_s": duration,
        "width": video_stream.get("width", ""),
        "height": video_stream.get("height", ""),
        "fps": fps,
        "nb_frames": nb_frames,
        "error": "",
    }


def main():
    mp4_files = [
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(".mp4")
    ]
    total = len(mp4_files)
    print(f"共发现 {total} 个 MP4 文件，使用 {NUM_WORKERS} 线程处理...")

    fieldnames = [
        "filepath", "filename", "size_mb", "duration_s",
        "width", "height", "fps", "nb_frames", "error"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(probe_video, fp): fp for fp in mp4_files}
            for i, future in enumerate(tqdm(as_completed(futures), total=total, desc="扫描进度")):
                row = future.result()
                for key in fieldnames:
                    row.setdefault(key, "")
                writer.writerow(row)
                if i % 100 == 0:
                    f.flush()

    print(f"\n完成！结果已写入: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
