<div align="center">

## Quick Start

> The inference environment is identical to Self Forcing, so you can migrate directly using our configs and model.

### Installation
```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Training

<details>
<summary> Stage 1: Autoregressive Diffusion Training </summary>

Then train the AR-diffusion model:
- Framewise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_framewise.yaml \
    --logdir logs/ar_diffusion_framewise
  ```

- Chunkwise:
  ```bash
    torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/ar_diffusion_tf_chunkwise.yaml \
    --logdir logs/ar_diffusion_chunkwise
  ```

> We recommend training no less than 8K steps, and more steps (e.g., 10~20K) will lead to better performance.


</details>


<details>
<summary> Stage 2: Causal ODE Initialization </summary>

In this stage, first generate ODE paired data:
```bash
# for the frame-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_framewise.py \
  --generator_ckpt checkpoints/framewise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_framewise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_framewise_latents \
  --lmdb_path dataset/ODE6KCausal_framewise

# for the chunk-wise model
torchrun --nproc_per_node=8 \
  get_causal_ode_data_chunkwise.py \
  --generator_ckpt checkpoints/chunkwise/ar_diffusion.pt \
  --rawdata_path dataset/clean_data \
  --output_folder dataset/ODE6KCausal_chunkwise_latents

python utils/create_lmdb_iterative.py \
  --data_path dataset/ODE6KCausal_chunkwise_latents \
  --lmdb_path dataset/ODE6KCausal_chunkwise
```

And then train ODE initialization models:
- Frame-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_framewise.yaml \
    --logdir logs/causal_ode_framewise
  ```
- Chunk-wise:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_ode_chunkwise.yaml \
    --logdir logs/causal_ode_chunkwise
  ```

> We recommend training no less than 10K steps, and more steps (e.g., 10~20K) will lead to better performance.

</details>

### Stage 3: DMD

> This stage is compatible with Self Forcing training, so you can migrate seamlessly by using our configs and checkpoints.

> Set your wandb configs before training.

And then train DMD models:

- Frame-wise model:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_framewise.yaml \
    --logdir logs/causal_forcing_dmd_framewise
  ```
  > We recommend training 500 steps. More than 1K steps will reduce dynamic degree.

- Chunk-wise model:
  ```bash
  torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/causal_forcing_dmd_chunkwise.yaml \
    --logdir logs/causal_forcing_dmd_chunkwise
  ```
  > We recommend training 200~300 steps. More than 1K steps will reduce dynamic degree.

Such models are the final models used to generate videos.
## FAQ & Blog 

**1. Why using bidirectional teacher in the DMD stage ?**
- Q: In the DMD stage, do you still use a bidirectional teacher? Why not an AR teacher?
- A: Yes. DMD only requires the student to match the teacher’s final distribution, not the generation trajectory, so a bidirectional teacher is fine. Also, bidirectional diffusion models are typically stronger than AR diffusion, so they make a better teacher.

- Q: Then why must the ODE (or Consistency Distillation) stage use an AR teacher?
- A: Because ODE/CD requires the student and teacher to follow the same trajectory, so their structures must be matched; an AR student cannot be trajectory-aligned with a bidirectional teacher.

**2. ODE initialization or multi-step AR diffusion initialization ?**
- Q: Which is better as initialization: a “proper” ODE initialization or directly using multi-step AR diffusion?
- A: We compared this in the Appendix C2. Overall, proper ODE init is better: multi-step AR diffusion init + DMD occasionally yields grid-like or waxy/greasy results. A key reason is that DMD is inherently few-step, so the right comparison is under few-step; in that regime, a few-step diffusion teacher is much weaker than an ODE-distilled teacher. Without ODE distillation, DMD must both close the step gap and handle an added conditioning gap from self-rollout: early few-step errors corrupt the history and get amplified across frames (large exposure bias), which increases DMD pressure. It can still converge, but typically with worse quality than ODE initialization. Also, with ODE init, DMD can be trained very few steps (e.g., ~100), reducing the risk of dynamics degradation from long DMD training.

**3. Can frame-level non-injectivity appears in the actual training dataset ?**
- Q: Regarding the “one-to-many” analysis in the ODE stage: since a single frame’s latent has very high dimensionality, isn’t the probability of being exactly identical extremely small?
- A: Yes, but the key point here is not whether the dataset literally contains identical samples; it’s whether there exists a well-defined function in the mathematical sense. Our vision modalities live in a continuous space—even in 1D, getting two samples to be exactly identical is extremely unlikely. However, the theoretical existence of exact collisions is enough to break the function property and make it ill-defined.

For more details, see [here](https://zhuanlan.zhihu.com/p/2002114039493461457). (currently in Chinese)

