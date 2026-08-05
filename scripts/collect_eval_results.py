#!/usr/bin/env python3
"""
收集 videos/ 下所有 *eval_results*.json 的第一级数值，整理成 CSV。
每行对应一个模型，四个数据集的结果展开为带前缀的列。
列顺序：model | {dataset}_quality_score | {dataset}_semantic_score | {dataset}_final_score | {dataset}_{dim} ...
"""

import csv
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

VIDEOS_DIR = Path(__file__).parent.parent / "videos"
OUTPUT_CSV  = VIDEOS_DIR / "eval_results_summary.csv"

DATASETS = [
    "all_dimension_extended-rename",
    "demos",
    "20260209170323-gemini",
    "20260209170323-vbench",
]

DS_DISPLAY = {
    "all_dimension_extended-rename": "all_dimension_extended-rename",
    "demos":                         "demos",
    "20260209170323-gemini":         "gemini",
    "20260209170323-vbench":         "vbench",
}

# ---------- VBench 常量 ----------
DIM_WEIGHT = {
    "subject consistency": 1, "background consistency": 1,
    "temporal flickering": 1, "motion smoothness": 1,
    "aesthetic quality": 1,   "imaging quality": 1,
    "dynamic degree": 0.5,    "object class": 1,
    "multiple objects": 1,    "human action": 1,
    "color": 1,               "spatial relationship": 1,
    "scene": 1,               "appearance style": 1,
    "temporal style": 1,      "overall consistency": 1,
}

NORMALIZE_DIC = {
    "subject consistency":    {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering":    {"Min": 0.6293, "Max": 1.0},
    "motion smoothness":      {"Min": 0.706,  "Max": 0.9975},
    "dynamic degree":         {"Min": 0.0,    "Max": 1.0},
    "aesthetic quality":      {"Min": 0.0,    "Max": 1.0},
    "imaging quality":        {"Min": 0.0,    "Max": 1.0},
    "object class":           {"Min": 0.0,    "Max": 1.0},
    "multiple objects":       {"Min": 0.0,    "Max": 1.0},
    "human action":           {"Min": 0.0,    "Max": 1.0},
    "color":                  {"Min": 0.0,    "Max": 1.0},
    "spatial relationship":   {"Min": 0.0,    "Max": 1.0},
    "scene":                  {"Min": 0.0,    "Max": 0.8222},
    "appearance style":       {"Min": 0.0009, "Max": 0.2855},
    "temporal style":         {"Min": 0.0,    "Max": 0.364},
    "overall consistency":    {"Min": 0.0,    "Max": 0.364},
}

QUALITY_LIST  = ["subject consistency", "background consistency", "temporal flickering",
                 "motion smoothness", "aesthetic quality", "imaging quality", "dynamic degree"]
SEMANTIC_LIST = ["object class", "multiple objects", "human action", "color",
                 "spatial relationship", "scene", "appearance style", "temporal style",
                 "overall consistency"]
QUALITY_WEIGHT, SEMANTIC_WEIGHT = 4, 1
EVAL_VARIANT_PATTERN = re.compile(r"(?:^|[-_])(ema|noema)_eval_results$")


def compute_scores(raw_values: dict):
    data = {k.replace("_", " "): v for k, v in raw_values.items()}
    normalized = {}
    for key in DIM_WEIGHT:
        if key not in data:
            continue
        mn, mx = NORMALIZE_DIC[key]["Min"], NORMALIZE_DIC[key]["Max"]
        normalized[key] = (data[key] - mn) / (mx - mn) * DIM_WEIGHT[key]

    def weighted_avg(keys):
        vals    = [normalized[k] for k in keys if k in normalized]
        weights = [DIM_WEIGHT[k] for k in keys if k in normalized]
        return sum(vals) / sum(weights) if weights else 0.0

    quality  = weighted_avg(QUALITY_LIST)
    semantic = weighted_avg(SEMANTIC_LIST)
    final    = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)
    return quality, semantic, final


def load_json(jf: Path):
    """返回 {dim_key: float} 以及计算好的三个分数。"""
    with open(jf) as f:
        data = json.load(f)
    raw = {k: (v[0] if isinstance(v, list) else v) for k, v in data.items()}
    quality, semantic, final = compute_scores(raw)
    return raw, quality, semantic, final


def get_model_name(rel_path: Path) -> str:
    """根据相对路径生成模型名，必要时附加 -ema/-noema 后缀。"""
    model_name = rel_path.parts[0]
    match = EVAL_VARIANT_PATTERN.search(rel_path.stem)
    if match:
        model_name = f"{model_name}-{match.group(1)}"
    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 videos/ 下的评测结果 JSON 为 CSV。")
    parser.add_argument(
        "--model-prefix",
        help="只统计模型名以该前缀开头的结果，例如 causal_forcing_dmd_chunkwise",
    )
    return parser.parse_args()


def collect(model_prefix: Optional[str] = None):
    json_files = sorted(VIDEOS_DIR.rglob("*eval_results*.json"))
    if not json_files:
        print("未找到任何 eval_results JSON 文件。")
        sys.exit(1)

    # model -> dataset -> (raw_values, quality, semantic, final)
    model_data: Dict[str, dict] = defaultdict(dict)
    # 每个数据集自己的维度列表（保序、去重）
    dataset_dim_keys: Dict[str, List[str]] = defaultdict(list)

    for jf in json_files:
        rel = jf.relative_to(VIDEOS_DIR)
        model_name   = get_model_name(rel)
        dataset_name = rel.parts[1] if len(rel.parts) > 2 else ""
        if model_prefix and not model_name.startswith(model_prefix):
            continue

        raw, quality, semantic, final = load_json(jf)
        model_data[model_name][dataset_name] = (raw, quality, semantic, final)

        for k in raw:
            if k not in dataset_dim_keys[dataset_name]:
                dataset_dim_keys[dataset_name].append(k)

    if not model_data:
        print(f"未找到模型名前缀为 {model_prefix!r} 的 eval_results JSON 文件。")
        sys.exit(1)

    # 判断每个数据集是否含语义维度
    semantic_dim_set = set(k.replace(" ", "_") for k in SEMANTIC_LIST)
    ds_has_semantic = {
        ds: bool(set(dataset_dim_keys[ds]) & semantic_dim_set)
        for ds in dataset_dim_keys
    }

    def col(ds, suffix):
        return f"{DS_DISPLAY[ds]}_{suffix}"

    # 构建列名：每个数据集先放分数（无语义维度则只放 quality_score），再放各维度
    fieldnames = ["model", "missing_datasets", "combined_final_score"]
    for ds in DATASETS:
        if ds in dataset_dim_keys:
            fieldnames.append(col(ds, "final_score"))
    for ds in DATASETS:
        if ds not in dataset_dim_keys:
            continue
        fieldnames.append(col(ds, "quality_score"))
        if ds_has_semantic[ds]:
            fieldnames.append(col(ds, "semantic_score"))
        for dim in dataset_dim_keys[ds]:
            fieldnames.append(col(ds, dim))

    rows = []
    for model_name in sorted(model_data):
        missing = [DS_DISPLAY[ds] for ds in DATASETS if ds not in model_data[model_name]]
        row = {"model": model_name, "missing_datasets": "; ".join(missing)}
        for ds in DATASETS:
            if ds not in model_data[model_name]:
                continue
            raw, quality, semantic, final = model_data[model_name][ds]
            row[col(ds, "quality_score")] = f"{quality * 100:.3f}"
            if ds_has_semantic[ds]:
                row[col(ds, "semantic_score")] = f"{semantic * 100:.3f}"
            row[col(ds, "final_score")] = f"{final * 100:.3f}"
            for dim, val in raw.items():
                row[col(ds, dim)] = f"{val * 100:.3f}" if isinstance(val, float) else val

        # 合成 final_score：四个数据集按 6:3:1:1 加权
        ds_weights = {
            "all_dimension_extended-rename": 6,
            "demos":                         3,
            "20260209170323-gemini":         1,
            "20260209170323-vbench":         1,
        }
        if all(row.get(col(ds, "final_score"), "") != "" for ds in ds_weights):
            weighted_sum = sum(float(row[col(ds, "final_score")]) * w for ds, w in ds_weights.items())
            row["combined_final_score"] = f"{weighted_sum / sum(ds_weights.values()):.3f}"

        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"共处理 {len(rows)} 个模型，结果写入：{OUTPUT_CSV}")


if __name__ == "__main__":
    args = parse_args()
    collect(model_prefix="causal_forcing_dmd_chunkwise")
