from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.wan_wrapper_22 import Wan22_DiffusionWrapper, Wan22_VAEWrapper
from utils.scheduler import FlowMatchScheduler
from utils.distributed import launch_distributed_job
from utils.global_config import set_wan_version, get_wan_version
from utils.dataset import LatentLMDBDataset, TextVideoDataset, I2VDataset

import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

def init_model(device, args, require_timesteps_idx=[0, 12, 24, 36]):
    set_wan_version(args, "init_model in get_causal_ode_data_chunkwise.py")
    if get_wan_version("init_model in get_causal_ode_data_chunkwise.py") == "2.2":
        _DiffusionWrapper = Wan22_DiffusionWrapper
        _VAEWrapper = Wan22_VAEWrapper
    else:
        _DiffusionWrapper = WanDiffusionWrapper
        _VAEWrapper = WanVAEWrapper
    
    # model = WanDiffusionWrapper(is_causal=True).to(device).to(torch.float32)
    model = _DiffusionWrapper(**getattr(args, "model_kwargs", {}),is_causal=True).to(device).to(torch.float32)
    model.model.num_frame_per_block = 4 # !!
    model.model.independent_first_frame = True

    vae_encoder = _VAEWrapper().to(device).to(torch.float32)
    text_encoder = WanTextEncoder().to(device).to(torch.float32)

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)
    
    def shift_time_steps(ts):
        s = float(scheduler.shift)
        T = float(scheduler.num_train_timesteps)
        t = torch.tensor(ts, dtype=torch.float32) / T
        return s * t / (1 + (s - 1) * t) * T
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"scheduler.timesteps: {scheduler.timesteps}")
    assert torch.equal(scheduler.timesteps[require_timesteps_idx], shift_time_steps([1000, 750, 500, 250]))

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = text_encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, vae_encoder, text_encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--rawdata_path", type=str)
    parser.add_argument("--generator_ckpt", type=str)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--require_timesteps_idx", type=int, nargs="+", default=[0, 12, 24, 36])
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--prefix", type=int, default=0,
                        help="Index prefix in millions, e.g. --prefix 1 makes prompt_index=2 become 1000002")

    args = parser.parse_args()

    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    model, vae_encoder, text_encoder, scheduler, unconditional_dict = init_model(device=device, args=config, require_timesteps_idx=args.require_timesteps_idx)
    state_dict = torch.load(args.generator_ckpt, map_location="cpu")
        
    gen_sd = state_dict["generator"]
    fixed = {}
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        fixed[k] = v
    state_dict = fixed
    model.model.load_state_dict(
        state_dict, strict=True
    )

    # dataset = LatentLMDBDataset(args.rawdata_path)
    if os.path.isdir(args.rawdata_path):
        dataset = LatentLMDBDataset(args.rawdata_path, max_pair=int(1e8))
    elif os.path.isfile(args.rawdata_path) and ".csv" in args.rawdata_path:
        vae_stride = 8 if get_wan_version("get_causal_ode_data_chunkwise.py") == "2.1" else 16
        dataset = I2VDataset(
            csv_path=args.rawdata_path,
            height=list(config.image_or_video_shape)[-2] * vae_stride,
            width=list(config.image_or_video_shape)[-1] * vae_stride,
        )
    else:
        raise NotImplementedError

    if args.output_folder is None:
        args.output_folder = os.path.join(
            "datasets/ODE_Causal_chunkwise_latents",
            # "/m2v_intern3/zhangjiaming09/Video-Causal-Dataset/ODE_Causal_chunkwise_latents",
            os.path.basename(args.config_path)[:-5], "-".join(args.generator_ckpt.split('/')[-3:-1]) + f"-gc-{args.guidance_scale}", os.path.basename(args.rawdata_path)[:-4]
        )
    print(f"args.output_folder: {args.output_folder}")
    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

    total_steps = int(math.ceil(len(dataset) / dist.get_world_size()))
    for index in tqdm(
        range(total_steps), disable=(int(os.environ.get("LOCAL_RANK", 0)) != 0),
    ):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue

        output_path = os.path.join(args.output_folder, f"{prompt_index + args.prefix * 100000000:08d}.pt")
        if os.path.exists(output_path):
            print(f"Skipping ODE pair {prompt_index} in rank {dist.get_rank()} (already exists)")
            continue

        print(f"Creating ODE pair {prompt_index} in rank {dist.get_rank()}")
        sample = dataset[prompt_index]
        prompt = sample["prompts"]
       
        # clean_latent = sample["clean_latent"].to(device).unsqueeze(0)
        if isinstance(dataset, LatentLMDBDataset):
            clean_latent = sample["clean_latent"].to(device).unsqueeze(0)
        elif isinstance(dataset, I2VDataset):
            frames = sample["frames"].to(device).unsqueeze(0)
            frames_vae_input = frames.permute(0, 2, 1, 3, 4).contiguous()
            clean_latent = vae_encoder.encode_to_latent(frames_vae_input).to(device)
        else:
            raise NotImplementedError
        assert list(clean_latent.shape) == config.image_or_video_shape

        conditional_dict = text_encoder(
            text_prompts=prompt
        )

        latents = torch.randn(
            list(config.image_or_video_shape), dtype=torch.float32, device=device
        )
        
        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps, disable=(int(os.environ.get("LOCAL_RANK", 0)) != 0), desc=f"rank{dist.get_rank()} idx{prompt_index}")):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)
            # timestep[:, 0] = 0 这里有bug，生成不了
            # if global_rank == 0:
            #     print('timestep: ',timestep)
            
            noisy_input.append(latents)

            f_cond, x0_pred_cond = model(
                latents, conditional_dict, timestep, clean_x = clean_latent
            )

            f_uncond, x0_pred_uncond = model(
                latents, unconditional_dict, timestep, clean_x = clean_latent
            )

            flow_pred = f_uncond + args.guidance_scale * (
                f_cond - f_uncond
            )
            
            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)
        noisy_input.append(clean_latent)
        
        noisy_inputs = torch.stack(noisy_input, dim=1)  # [len(scheduler.timesteps) + 1(x_0) + 1(clean)]

        noisy_inputs = noisy_inputs[:, args.require_timesteps_idx + [-2, -1]] # 4 ODE + 1(x_0) + 1(clean)

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            output_path
        )

        # if global_rank == 0:
        #     final_latent = noisy_inputs[:, -2, ...].to(device=device, dtype=torch.float32)
        #     print('final_latent shape: ',final_latent.shape)
        #     videos = vae_encoder.decode_to_pixel(final_latent)
        #     videos = (videos * 0.5 + 0.5).clamp(0, 1)
        #     videos = rearrange(videos, 'b t c h w -> b t h w c').cpu()
        #     videos = 255.0 * videos[0]

        #     vis_dir = os.path.join("vis", "ode_examples")
        #     os.makedirs(vis_dir, exist_ok=True)
        #     vis_path = os.path.join(vis_dir, f"{prompt_index + args.prefix * 100000000:08d}.mp4")
        #     write_video(vis_path, videos, fps=16)

    dist.barrier()


if __name__ == "__main__":
    main()
