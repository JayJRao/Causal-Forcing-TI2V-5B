import gc
import logging

from model import CausalDiffusion
from utils.dataset import cycle, LatentLMDBDataset, TextVideoDataset, TextDataset, I2VDataset, I2VDataset_with_mask_pose
from utils.build_object_hand_regions import build_object_hand_regions
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.global_config import get_wan_version

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        print(f"global_rank: {global_rank}, self.world_size: {self.world_size}, self.is_main_process: {self.is_main_process}")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if getattr(config, "i2v", False):
            vae_stride = 8 if get_wan_version(str(self.__class__)) == "2.1" else 16
            dataset = I2VDataset(
                csv_path=config.data_path,
                height=list(self.config.image_or_video_shape)[-2] * vae_stride,
                width=list(self.config.image_or_video_shape)[-1] * vae_stride,
            )
            dataset = I2VDataset_with_mask_pose(
                config.data_path,
                height=1280,
                width=704,
                prompt_col="caption",
                max_items=70000,
                return_video=True,
                return_image=True,
                return_mask_video=True,
                return_pose_video=True,
            )
        elif os.path.isdir(config.data_path) and not config.load_raw_video:
            dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
        elif os.path.isfile(config.data_path) and ".csv" in config.data_path:
            vae_stride = 8 if get_wan_version(str(self.__class__)) == "2.1" else 16
            dataset = TextVideoDataset(
                csv_path=config.data_path,
                height=list(self.config.image_or_video_shape)[-2] * vae_stride,
                width=list(self.config.image_or_video_shape)[-1] * vae_stride,
            )
        else:
            raise NotImplementedError
       
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=0 if isinstance(dataset, LatentLMDBDataset) else 8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)
            self.step = 13001

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.eval_interval = getattr(self.config, "eval_interval", 1000)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            assert not hasattr(config, 'denoising_step_list')
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device,
                            generator=self.model.generator, text_encoder=self.model.text_encoder, vae=self.model.vae)
            # self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            # self.pipeline.generator = self.model.generator
            # self.pipeline.text_encoder = self.model.text_encoder
            # self.eval_dataset = TextDataset(prompt_path="prompts/eval.txt")
        self.hand_weight = getattr(self.config, "hand_weight", 1)
        self.object_weight = getattr(self.config, "object_weight", 1)
        
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        object_region_21 = None
        hand_region_21 = None

        if getattr(self.config, "i2v", False):
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            frames_vae_input = frames.permute(0, 2, 1, 3, 4).contiguous()
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames_vae_input).to(device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1, ]

            mask_video = batch.get("mask_video", None)
            pose_video = batch.get("pose_video", None)
            if mask_video is not None:
                mask_video = mask_video.to(device=self.device)
            if pose_video is not None:
                pose_video = pose_video.to(device=self.device)

            if mask_video is not None and pose_video is not None:
                object_region_21, hand_region_21 = build_object_hand_regions(
                    mask_video=mask_video,
                    pose_video=pose_video,
                    clean_latent=clean_latent
                )
                print('object_region_21.shape: ',object_region_21.shape)
                print('hand_region_21.shape: ',hand_region_21.shape)

        elif not self.config.load_raw_video:
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1, ]
        else:
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1, ]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts) 
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            object_region=object_region_21,
            hand_region=hand_region_21,
            object_weight=self.object_weight,
            hand_weight=self.hand_weight,
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            
            current_time = time.time()
            iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
            print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_loss.mean().item()}, generator_grad_norm {generator_grad_norm.mean().item()}")
            self.previous_time = current_time

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    # =========================
    # Parallel "eval": mean tensor of (pred_latent - clean_latent) on predicted segment
    # pred has num_output_frames frames, and pred[:num_init_frames] == initial_latent (guaranteed)
    # =========================
    @torch.no_grad()
    def _generate_latents(self, prompts, *, initial_latent=None, num_output_frames=21, num_init_frames=3):
        bsz = len(prompts)
        device = torch.device("cuda", torch.cuda.current_device())
        noise_T = num_output_frames if initial_latent is None else (num_output_frames - num_init_frames)
        # noise = torch.randn([bsz, noise_T, 16, 60, 104], device=device, dtype=self.dtype)
        image_or_video_shape = list(self.config.image_or_video_shape)
        noise = torch.randn([bsz, noise_T, image_or_video_shape[-3], image_or_video_shape[-2], image_or_video_shape[-1]], device=device, dtype=self.dtype)

        latents = self.pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            return_video=False
        )
        return latents  # expected: [B, num_output_frames, 16, 60, 104]


    @torch.no_grad()
    def parallel_visualization(self, rank0_index, *, num_output_frames=21, num_init_frames=3):
        prompt_index = int(rank0_index) + dist.get_rank()
        if prompt_index < len(self.dataset):
            sample = self.dataset[prompt_index]
            prompt = sample["prompts"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            
            # clean = sample["clean_latent"].to(device=device, dtype=self.dtype)  # [>=21,16,60,104]
            # clean = clean[:num_output_frames]
            if getattr(self.config, "i2v", False):
                frames = sample["frames"].to(device=self.device, dtype=self.dtype)
                frames_vae_input = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
                with torch.no_grad():
                    clean_latent = self.model.vae.encode_to_latent(
                        frames_vae_input).to(device=self.device, dtype=self.dtype)
                clean_latent = clean_latent.squeeze(0)
            elif not self.config.load_raw_video:
                clean_latent = sample["clean_latent"].to(
                    device=self.device, dtype=self.dtype)
            else:
                frames = sample["frames"].to(
                    device=self.device, dtype=self.dtype)
                frames = frames.unsqueeze(0)
                with torch.no_grad():
                    clean_latent = self.model.vae.encode_to_latent(
                        frames).to(device=self.device, dtype=self.dtype)
                clean_latent = clean_latent.squeeze(0)
            clean = clean_latent[:num_output_frames]
            print(f'clean.shape is {clean.shape}')
            init_latent = None
            
            if self.given_first_chunk:
                init_latent = clean[:num_init_frames].unsqueeze(0)

            latents = self._generate_latents(
                [prompt],
                initial_latent=init_latent,
                num_output_frames=num_output_frames,
                num_init_frames=num_init_frames
            )

            from torchvision.io import write_video
            from einops import rearrange

            videos = self.model.vae.decode_to_pixel(latents.to(device=torch.cuda.current_device(), dtype=torch.bfloat16))
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, 'b t c h w -> b t h w c').cpu()
            videos = 255.0 * videos[0]
            video_name = f"step_{self.step:06d}_rank_{dist.get_rank()}_prompt_{prompt_index}.mp4"
            out_path = os.path.join(
                f'vis/{self.config.config_name}',
                video_name,
            )
            os.makedirs(f'vis/{self.config.config_name}', exist_ok=True)
            write_video(out_path, videos, fps=16)

            clean_video = self.model.vae.decode_to_pixel(clean_latent.unsqueeze(0).to(device=torch.cuda.current_device(), dtype=torch.bfloat16))
            clean_video = (clean_video * 0.5 + 0.5).clamp(0, 1)
            clean_video = rearrange(clean_video, 'b t c h w -> b t h w c').cpu()
            clean_video = 255.0 * clean_video[0]
            video_name = f"step_{self.step:06d}_rank_{dist.get_rank()}_prompt_{prompt_index}_clean.mp4"
            out_path = os.path.join(
                f'vis/{self.config.config_name}',
                video_name,
            )
            os.makedirs(f'vis/{self.config.config_name}', exist_ok=True)
            write_video(out_path, clean_video, fps=16)

            if hasattr(self.model, "vae") and hasattr(self.model.vae, "model") and hasattr(self.model.vae.model, "clear_cache"):
                self.model.vae.model.clear_cache()

    @torch.no_grad()
    def parallel_visualization_without_init(self, rank0_index, *, num_output_frames=21):
        prompt_index = int(rank0_index) + dist.get_rank()
        if prompt_index < len(self.eval_dataset):
            sample = self.eval_dataset[prompt_index]
            prompt = sample["prompts"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]

            latents = self._generate_latents(
                [prompt],
                initial_latent=None,
                num_output_frames=num_output_frames,
                num_init_frames=0
            )

            from torchvision.io import write_video
            from einops import rearrange

            videos = self.model.vae.decode_to_pixel(latents.to(device=torch.cuda.current_device(), dtype=torch.bfloat16))
            videos = (videos * 0.5 + 0.5).clamp(0, 1)
            videos = rearrange(videos, 'b t c h w -> b t h w c').cpu()
            videos = 255.0 * videos[0]
            video_name = f"step_{self.step:06d}_rank_{dist.get_rank()}_prompt_{prompt_index}_output_{num_output_frames}_without_init.mp4"
            out_path = os.path.join(
                f'vis/{self.config.config_name}',
                video_name,
            )
            os.makedirs(f'vis/{self.config.config_name}', exist_ok=True)
            write_video(out_path, videos, fps=16)

            if hasattr(self.model, "vae") and hasattr(self.model.vae, "model") and hasattr(self.model.vae.model, "clear_cache"):
                self.model.vae.model.clear_cache()

    def train(self):

        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.eval_interval and (self.step % self.eval_interval == 0):
                was_train_gen = self.model.generator.training
                was_train_txt = self.model.text_encoder.training
                self.model.generator.eval()
                self.model.text_encoder.eval()

                if dist.get_rank() == 0:
                    idx_t = torch.randint(0, len(self.dataset) - self.world_size * 2, (1,), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
                else:
                    idx_t = torch.zeros((1,), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
                dist.broadcast(idx_t, src=0)
                index = int(idx_t.item())

                self.parallel_visualization(
                    rank0_index=index,
                    num_output_frames=self.eval_frames,
                    num_init_frames=self.eval_init
                )
                # if self.step % (self.eval_interval * 2) == 0:
                #     self.parallel_visualization_without_init(
                #         rank0_index=0,
                #         num_output_frames=self.eval_frames * 3,
                #         # num_init_frames=self.eval_init
                #     )

                if was_train_gen:
                    self.model.generator.train()
                if was_train_txt:
                    self.model.text_encoder.train()

            barrier()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
