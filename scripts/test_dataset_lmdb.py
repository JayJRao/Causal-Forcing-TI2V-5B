"""
全量读取测试：检查 LatentLMDBDataset 中的损坏数据
用法：
    python test_dataset.py --data_path /ytech_milm_intern/data_share/Causal-Forcing-data/clean_data
    python test_dataset.py --data_path /path/to/lmdb --num_workers 0  # 单进程，便于定位 segfault
"""
import argparse
import traceback
import sys
import time
import torch
from torch.utils.data import DataLoader

# 把项目根目录加入 path，确保能 import utils
sys.path.insert(0, "/m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing")
from utils.dataset import LatentLMDBDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/ytech_milm_intern/data_share/Causal-Forcing-data/clean_data")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers；设为 0 可用单进程定位 segfault")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=500,
                        help="每隔多少 batch 打印一次进度")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[init] 打开数据集: {args.data_path}")
    dataset = LatentLMDBDataset(args.data_path, max_pair=int(1e8))
    total = len(dataset)
    print(f"[init] 数据集大小: {total} 条")
    print(f"[init] latents shape（单条）: {dataset.latents_shape[1:]}")
    print(f"[init] num_workers={args.num_workers}, batch_size={args.batch_size}")
    print("-" * 60)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    bad_indices = []
    total_batches = (total + args.batch_size - 1) // args.batch_size
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        # 基本结构检查
        if "clean_latent" not in batch or "prompts" not in batch:
            print(f"[ERROR] batch {batch_idx}: 缺少必要字段，keys={list(batch.keys())}")
            bad_indices.append(batch_idx * args.batch_size)
            continue

        latent = batch["clean_latent"]
        prompts = batch["prompts"]

        # 检查 NaN / Inf
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            start_idx = batch_idx * args.batch_size
            print(f"[WARN ] batch {batch_idx} (idx {start_idx}~{start_idx + len(prompts) - 1}): "
                  f"latent 含有 NaN/Inf")
            bad_indices.extend(range(start_idx, start_idx + len(prompts)))

        # 进度日志
        if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            done = (batch_idx + 1) * args.batch_size
            speed = done / elapsed
            eta = (total - done) / speed if speed > 0 else 0
            print(f"[{done:>7}/{total}]  bad={len(bad_indices)}  "
                  f"speed={speed:.1f} samples/s  ETA={eta:.0f}s")

    print("-" * 60)
    elapsed = time.time() - t0
    print(f"[done] 共扫描 {total} 条，耗时 {elapsed:.1f}s")
    if bad_indices:
        print(f"[done] 发现 {len(bad_indices)} 条异常数据，起始索引：{bad_indices[:20]}"
              f"{'...' if len(bad_indices) > 20 else ''}")
    else:
        print("[done] 未发现 NaN/Inf 等数据异常")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
