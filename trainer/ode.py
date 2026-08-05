import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle
from model import ODERegression
from collections import defaultdict
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)


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
        self.disable_wandb = config.disable_wandb
        print(f"global_rank: {global_rank}, self.world_size: {self.world_size}, self.is_main_process: {self.is_main_process}")

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
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
        self.model = ODERegression(config, device=self.device)
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
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
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
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        self.dataset = dataset

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)


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

        ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_time = None

        if not config.no_visualize:
            assert hasattr(config, 'denoising_step_list')
            self.pipeline = CausalInferencePipeline(config, device=self.device,
                            generator=self.model.generator, text_encoder=self.model.text_encoder, vae=self.model.vae)
            self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
            self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
            self.eval_init = getattr(self.config, "eval_num_init_frames", 3)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
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

    def train_one_step(self, batch, loss_scale=1.0):
        
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(
            device=self.device, dtype=self.dtype)
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        (generator_loss * loss_scale).backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        # Step 4: Logging
        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
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
        noise = torch.randn([bsz, noise_T, initial_latent.shape[-3], initial_latent.shape[-2], initial_latent.shape[-1]], device=device, dtype=self.dtype)

        _, latents = self.pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
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
            
            ode_latent = sample["ode_latent"].to(
                device=self.device, dtype=self.dtype)
            clean_latent = ode_latent[-1].unsqueeze(0)
            print(f'clean_latent.shape is {clean_latent.shape}')
            init_latent = None
            
            if self.given_first_chunk:
                init_latent = clean_latent[:, :num_init_frames]

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

            clean_video = self.model.vae.decode_to_pixel(clean_latent.to(device=torch.cuda.current_device(), dtype=torch.bfloat16))
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


    def train(self):

        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            
            if not self.config.no_visualize and (self.step % self.config.log_iters == 0):
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
