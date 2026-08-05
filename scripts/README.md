# Scripts 说明

本目录包含用于 koala36m 视频数据集分析与筛选的工具脚本。

---

## 1. `analyze_sample.py` — 数据采样分析（v1）

快速分析 `koala36m.csv` 前 N 行数据的分布情况。

**数据源**：`/m2v_intern/m2v_data/data_version/koala36m.csv`

**功能**：
- 视频方向分布（横向 / 纵向 / 方形）
- 分辨率 Top 10 分布
- 总帧数（`total_frame_num`）分布：百分位数、区间直方图、阈值统计
- 训练适合度评分（`video_training_suitability_score`）分布
- 视频时长分布
- 横向视频 × 帧数阈值 × 评分阈值的组合筛选数据量

**运行**：
```bash
python scripts/analyze_sample.py
```

---

## 2. `analyze_sample_v2.py` — 数据采样分析（v2）

`analyze_sample.py` 的升级版，支持更多评分字段，对应新版 CSV。

**数据源**：`/m2v_intern/m2v_data/data_version/koala36m_250124.csv`

**相比 v1 新增**：
- 清晰度评分（`clarity_score`）分布与阈值统计
- 美学评分（`aesthetic_score`）分布与阈值统计
- 运动评分（`motion_score`）分布与阈值统计
- 多评分组合筛选示例（训练分 × 清晰度 × 美学）

**运行**：
```bash
python scripts/analyze_sample_v2.py
```

---

## 3. `filter_videos.py` — 视频数据筛选工具（增强版）

命令行筛选工具，支持多维度条件过滤，分块读取大文件，内存高效。

**数据源**：`/m2v_intern/m2v_data/data_version/koala36m_250124.csv`（可通过 `--input` 指定）

**筛选参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--orientation` | `horizontal` | 视频方向：`horizontal` / `vertical` / `all` |
| `--frames-min` | `100` | 帧数下限（严格大于） |
| `--frames-max` | `150` | 帧数上限（小于等于） |
| `--train-min/max` | 不限 | 训练适合度评分范围 |
| `--clarity-min/max` | 不限 | 清晰度评分范围 |
| `--aesthetic-min/max` | 不限 | 美学评分范围 |
| `--motion-min/max` | 不限 | 运动评分范围 |
| `--check-files` | 否 | 多进程检查视频文件是否存在 |
| `--save` | 否 | 保存结果到 CSV |
| `--output` | 自动生成 | 输出文件路径（自动文件名含筛选参数） |
| `--workers` | CPU核心数(≤32) | 并行进程数 |

**输出文件命名规则**（自动生成时）：
```
koala36m_filtered_{方向}_{帧数范围}f_{各评分条件}_{结果数量}.csv
```

**示例**：
```bash
# 横向视频，90-120帧，训练>=4.0，清晰度>=0.7，美学>=4.0，运动>=50
python scripts/filter_videos.py \
    --orientation horizontal \
    --frames-min 90 \
    --frames-max 120 \
    --train-min 4.0 \
    --clarity-min 0.7 \
    --aesthetic-min 4.0 \
    --motion-min 50.0 \
    --save

# 高动态视频，运动范围 20-40
python scripts/filter_videos.py \
    --frames-min 100 --frames-max 150 \
    --train-min 4.0 \
    --motion-min 20 --motion-max 40 \
    --save

# 同时检查视频文件是否存在
python scripts/filter_videos.py \
    --train-min 4.0 --clarity-min 0.6 \
    --check-files --save
```

---

## 4. `merge_csvs.py` — CSV 数据集融合工具

将 `datasets/` 下的多个 CSV 文件按指定比例融合为一个数据集。

**功能**：
- 支持任意数量的 CSV 文件，比例可为任意正数（相对值）
- 不指定总量时，自动以瓶颈文件为上限，最大化无放回采样量
- 指定 `--total` 时，按比例分配目标行数
- 默认随机打乱输出顺序，随机种子可控

**参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--files` | 必填 | 输入 CSV 文件路径列表 |
| `--ratios` | 必填 | 各文件对应的比例权重（相对值） |
| `--total` | 自动 | 期望输出总行数（不指定则自动取最大可用量） |
| `--output` | 自动生成 | 输出文件路径 |
| `--seed` | `42` | 随机种子 |
| `--no-shuffle` | 否 | 不打乱输出顺序 |

**输出文件命名规则**（自动生成时）：
```
merged_{YYYYMMDDHHMMSS}_{总行数}_{num1}x{ratio1}_{num2}x{ratio2}_.csv
```

**示例**：
```bash
# 两个文件按 1:1 融合（自动最大采样量）
python scripts/merge_csvs.py \
    --files datasets/koala36m_filtered_..._335860.csv \
            datasets/koala36m_filtered_..._183338.csv \
    --ratios 1 1

# 两个文件按 2:1 融合，指定总量 30 万条
python scripts/merge_csvs.py \
    --files datasets/a.csv datasets/b.csv \
    --ratios 2 1 \
    --total 300000

# 三个文件按 1:1:1 融合，自定义输出路径
python scripts/merge_csvs.py \
    --files datasets/a.csv datasets/b.csv datasets/c.csv \
    --ratios 1 1 1 \
    --output datasets/my_merged.csv
```

---

## 5. `test_dataset.py` — LMDB 数据集完整性检查

全量扫描 `LatentLMDBDataset`，检测其中的损坏数据（NaN/Inf latent、缺字段等）。

**参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_path` | `/ytech_milm_intern/data_share/Causal-Forcing-data/clean_data` | LMDB 数据集路径 |
| `--num_workers` | `4` | DataLoader worker 数（设为 0 可单进程定位 segfault）|
| `--batch_size` | `4` | 每批读取的样本数 |
| `--log_interval` | `500` | 每隔多少 batch 打印一次进度 |

**运行**：
```bash
# 默认路径
python scripts/test_dataset.py

# 指定路径，单进程（便于定位 segfault）
python scripts/test_dataset.py \
    --data_path /path/to/lmdb \
    --num_workers 0
```

---

## 6. `count_mp4.sh` — 统计各子目录 mp4 文件数量

遍历 `videos/` 目录的三级结构，统计每个叶目录中的 mp4 文件数量，并高亮显示数量异常（不符合预期值）的目录。

**预期数量规则**（内置）：

| 子目录名含 | 预期数量 |
|---|---|
| `demos*` | 100 |
| `all_dimension_extended*` | 946 |
| `*vbench*` | 30 |
| `*gemini*` | 20 |
| 其他 | 不检查 |

**用法**：
```bash
# 检查默认 videos/ 目录，只显示异常目录
bash scripts/count_mp4.sh

# 指定目录
bash scripts/count_mp4.sh /path/to/videos

# 显示所有目录（包括正常的）
bash scripts/count_mp4.sh videos -a
```

---

## 7. `rename_by_txt.py` — 按 txt 内容批量重命名 mp4

将 `videos/` 下所有 `all_dimension_extended` 叶目录中的 mp4，按 txt 文件中的行内容重命名，并复制到 `all_dimension_extended-rename` 目录下。txt 行数须与每个文件夹的 mp4 数量一致，按文件名排序后一一对应。

**参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `txt_path` | `prompts/vbench/all_dimension.txt` | 行数与每个文件夹 mp4 数量一致的 txt 文件 |
| `--videos-root` | `<repo>/videos` | videos 根目录 |
| `--dry-run` | 否 | 只打印操作，不实际复制 |
| `--limit` | 0（全部）| 只处理前 N 个文件夹 |

**运行**：
```bash
# 使用默认 txt，处理所有文件夹
python scripts/rename_by_txt.py

# 指定 txt 文件和 videos 根目录
python scripts/rename_by_txt.py prompts/my_prompts.txt \
    --videos-root /path/to/videos

# 预览（不实际操作）
python scripts/rename_by_txt.py --dry-run

# 只处理前 5 个文件夹
python scripts/rename_by_txt.py --limit 5
```

---

## 8. `collect_eval_results.py` — 汇总评测结果为 CSV

扫描 `videos/` 下所有 `*eval_results*.json` 文件，按 VBench 标准计算 quality_score、semantic_score、final_score，并将各模型在四个数据集上的结果汇总为一张 CSV 表格。

如果同一模型目录下存在 `...-ema_eval_results.json` 和 `...-noema_eval_results.json` 两类结果，脚本会分别将它们视为 `原模型名-ema` 和 `原模型名-noema` 两个独立模型，避免同一数据集结果互相覆盖。

**输出**：`videos/eval_results_summary.csv`

**四个数据集**（对应 `videos/<model>/` 下的子目录名）：
- `all_dimension_extended-rename`（权重 6）
- `demos`（权重 3）
- `20260209170323-gemini`（权重 1）
- `20260209170323-vbench`（权重 1）

最终 `combined_final_score` 为四个数据集按上述权重加权平均。

**运行**：
```bash
python scripts/collect_eval_results.py
```

只统计指定前缀的模型：
```bash
python scripts/collect_eval_results.py --model-prefix causal_forcing_dmd_chunkwise
```

---

## 9. `concat_videos_vertical.py` — 多模型视频纵向拼接对比

将多个模型在同一数据集下的同名视频纵向拼接为一个对比视频，每个视频上叠加模型名称标注及评测指标分数（左上角模型名，右上角各维度分数）。依赖 `ffmpeg`。

**参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--models` | 必填 | 模型文件夹名列表（至少 2 个）|
| `--dataset` | 必填 | 数据集子目录名 |
| `--videos_dir` | `videos` | videos 根目录 |
| `--output_dir` | 自动生成 | 输出目录（默认 `videos/concat_{dataset}_{时间戳}`）|
| `--width` | 自动取最小宽度 | 归一化目标宽度（像素）|
| `--video` | 无 | 只处理指定文件名的单个视频 |
| `--limit` | 无 | 只处理前 N 个视频（按文件名排序）|

**运行**：
```bash
# 三个模型在 all_dimension_extended 数据集上的对比
python scripts/concat_videos_vertical.py \
    --models modelA modelB modelC \
    --dataset all_dimension_extended

# 指定 videos 目录、输出目录和目标宽度
python scripts/concat_videos_vertical.py \
    --models modelA modelB \
    --dataset demos \
    --videos_dir /path/to/videos \
    --output_dir /path/to/output \
    --width 1280

# 只拼接单个视频
python scripts/concat_videos_vertical.py \
    --models modelA modelB \
    --dataset all_dimension_extended \
    --video "some_video-0.mp4"

# 只处理前 10 个视频
python scripts/concat_videos_vertical.py \
    --models modelA modelB \
    --dataset all_dimension_extended \
    --limit 10
```
