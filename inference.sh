GPU_ID="$1"             # e.g. 0,1,2,3
config_path="$2"        # e.g. configs/ar_diffusion_tf_chunkwise.yaml
checkpoint_path="$3"    # e.g. zhuhz22/Causal-Forcing/chunkwise/causal_forcing.pt or logs/ar_diffusion_tf_chunkwise_1.3b_koala36m_335860/checkpoint_model_010000/model.pt
data_path="$4"          # e.g. prompts/20260209170323-gemini.txt
num_output_frames="$5"  # e.g. 21
use_ema="$6"            # Y or N

# git 相关
GIT_VERSION=$(git rev-parse --short HEAD)

# python 相关
PYTHON_ENV_NAME=$(basename "$(dirname "$(dirname "$(which python)")")")

# gpu 相关
GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_ID")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
GPU_NAME=$(echo "$GPU_NAME" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')

# checkpoint_path 相关
if [[ "$checkpoint_path" =~ (^|.*/)_logs_v3/(.+)/checkpoint_model_([0-9]+)/model\.pt$ ]]; then
  gen_name="${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
elif [[ "$checkpoint_path" =~ (^|.*/)logs/(.+)/checkpoint_model_([0-9]+)/model\.pt$ ]]; then
  gen_name="${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
else
  gen_name="$(basename "$checkpoint_path" .pt)"
fi

# data_path 相关
data_file=$(basename "$data_path" .txt)

# use_ema 标签
EMA_TAG=$([ "$use_ema" = "Y" ] && echo "ema" || echo "noema")

# output文件夹
output_folder="videos/${gen_name}/${data_file}/${GIT_VERSION}-${PYTHON_ENV_NAME}-${GPU_NAME}-$(echo "$GPU_ID" | tr -d ',')-frames${num_output_frames}-${EMA_TAG}"
output_log="${output_folder}.log"
mkdir -p $output_folder

echo GPU_ID: $GPU_ID
echo config_path: $config_path
echo checkpoint_path: $checkpoint_path
echo data_path: $data_path
echo num_output_frames: $num_output_frames
echo use_ema: $use_ema
echo output_folder: $output_folder
echo output_log: $output_log

{
  echo "===== ENV: pip list ====="
  pip list
  echo "===== START torchrun ====="
} >> "$output_log" 2>&1

CUDA_VISIBLE_DEVICES=$GPU_ID NCCL_DEBUG=WARN torchrun \
  --nproc_per_node=$GPU_NUM \
  --master_port=$((29600 + $(echo "$GPU_ID" | cut -d',' -f1))) \
  inference.py \
  --config_path $config_path \
  --checkpoint_path $checkpoint_path \
  --data_path $data_path \
  --output_folder $output_folder \
  --num_output_frames $num_output_frames \
  $([ "$use_ema" = "Y" ] && echo "--use_ema") \
  2>&1 | tee -a "$output_log"

# bash /m2v_intern/zhangjiaming09/Video-Causal/input_gpu.sh $GPU_ID
