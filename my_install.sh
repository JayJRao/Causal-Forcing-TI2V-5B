# registry.corp.kuaishou.com/kml-supercomputing-project/jisihui-3.4-ft:m2v_nv_torch221_cu12_ema_0524-snapshot-18594-20250225193716-snapshot-18837-20250304014032

cd /m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing
source ~/.bashrc && conda deactivate && conda activate base && which python
export http_proxy=http://10.66.29.113:11080 https_proxy=http://10.66.29.113:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
conda create -n causal_forcing_251 python=3.11 -y
conda activate causal_forcing_251
which python && which pip && which conda && pip cache dir
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch;print(torch.__version__);print(torch.cuda.is_available())" && cd ../ && bash train.sh && cd - && pwd
pip install -r requirements.txt
pip install --force-reinstall pip==25.2 setuptools==80.10.2
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
pip install git+https://github.com/openai/CLIP.git
# pip install flash-attn --no-build-isolation
pip install /m2v_intern/zhangjiaming09/I2V_Animation/sota_animate/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl # pip install flash-attn==2.7.1.post4
python -c "import flash_attn" && pip list | grep flash
python setup.py develop


cd /m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing
source ~/.bashrc && conda deactivate && conda activate base && which python
export http_proxy=http://10.66.29.113:11080 https_proxy=http://10.66.29.113:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
conda create -n torch271128 python=3.11 -y
conda activate torch271128 && which python && pip cache dir
which python && which pip && which conda && pip cache dir
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__);print(torch.cuda.is_available())" && bash train.sh && pwd
pip install -r requirements.txt
pip install --force-reinstall pip==25.2 setuptools==80.10.2
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
pip install git+https://github.com/openai/CLIP.git
# pip install flash-attn --no-build-isolation
pip install /m2v_intern/zhangjiaming09/I2V_Animation/sota_animate/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl # pip install flash-attn==2.7.1.post4
python -c "import flash_attn" && pip list | grep flash
python setup.py develop


ln -s /m2v_intern/zhangjiaming09/models_ckpt/Wan2.1-T2V-1.3B/ /m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing/wan_models/Wan2.1-T2V-1.3B
ln -s /m2v_intern/zhangjiaming09/models_ckpt/Wan2.1-T2V-14B/ /m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing/wan_models/Wan2.1-T2V-14B
ln -s /m2v_intern/zhangjiaming09/models_ckpt/Wan2.2-TI2V-5B/ /m2v_intern/zhangjiaming09/Video-Causal/Causal-Forcing/wan_models/Wan2.2-TI2V-5B

################################################################################################################################

## Inference: T2V, Chunk-wise model
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder output/chunkwise \
  --checkpoint_path zhuhz22/Causal-Forcing/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt


################################################################################################################################


## Stage 1: Autoregressive Diffusion Training -> chunkwise/ar_diffusion.pt
# https://github.com/thu-ml/Causal-Forcing/issues/8
### 1. First download the dataset (we provide a 6K toy dataset here):
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset  # /ytech_milm_intern/data_share/Causal-Forcing-data
python utils/merge_and_get_clean.py

### 2. Then train the AR-diffusion model (Chunkwise):
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint $MASTER_ADDR \
train.py \
--config_path configs/ar_diffusion_tf_chunkwise.yaml \
--logdir logs/ar_diffusion_chunkwise


################################################################################################################################


## Stage 2: Causal ODE Initialization -> chunkwise/causal_ode.pt

### 1. In this stage, first generate ODE paired data:
torchrun --nproc_per_node=8 \
  get_causal_ode_data_chunkwise.py \
  --generator_ckpt checkpoints/chunkwise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_chunkwise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_chunkwise_latents \
  --lmdb_path dataset/ODE6KCausal_chunkwise

### 1. Or you can also directly download our prepared dataset (~300G):
hf download zhuhz22/Causal-Forcing-data  --local-dir dataset  # /ytech_milm_intern/data_share/Causal-Forcing-data
python utils/merge_lmdb.py

### 2. And then train ODE initialization models:
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/causal_ode_chunkwise.yaml \
  --logdir logs/causal_ode_chunkwise
