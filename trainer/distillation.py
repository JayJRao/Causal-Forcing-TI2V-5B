import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
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
        self.model = DMD(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", True),
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

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
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
            state_dict = torch.load(config.generator_ckpt, map_location="cpu", mmap=True)
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

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        if not config.no_visualize:
            assert hasattr(config, 'denoising_step_list')
            self.pipeline = CausalInferencePipeline(config, device=self.device,
                            generator=self.model.generator, text_encoder=self.model.text_encoder, vae=self.model.vae)
            # self.given_first_chunk = getattr(self.config, "given_first_chunk", False)
            self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
            # self.eval_init = getattr(self.config, "eval_num_init_frames", 0)
            self.eval_dataset = TextDataset(prompt_path="prompts/eval.txt")

    def save(self):
        if self.is_main_process:
            print(f"Start gathering distributed model states in step {self.step}...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "step": self.step,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                # "generator_ema": self.generator_ema.state_dict(),
                "step": self.step,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

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

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
           
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    # =========================
    # Parallel "eval": mean tensor of (pred_latent - clean_latent) on predicted segment
    # pred has num_output_frames frames, and pred[:num_init_frames] == initial_latent (guaranteed)
    # =========================
    @torch.no_grad()
    def _generate_latents(self, prompts, *, num_output_frames=21):
        bsz = len(prompts)
        device = torch.device("cuda", torch.cuda.current_device())
        # noise = torch.randn([bsz, noise_T, 16, 60, 104], device=device, dtype=self.dtype)
        image_or_video_shape = list(self.config.image_or_video_shape)
        noise = torch.randn([bsz, num_output_frames, image_or_video_shape[-3], image_or_video_shape[-2], image_or_video_shape[-1]], device=device, dtype=self.dtype)

        _, latents = self.pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=True,
        )
        return latents  # expected: [B, num_output_frames, 16, 60, 104]

    @torch.no_grad()
    def parallel_visualization_without_init(self, rank0_index, *, num_output_frames=21):
        prompt_index = int(rank0_index) + dist.get_rank()
        if prompt_index < len(self.eval_dataset):
            sample = self.eval_dataset[prompt_index]
            prompt = sample["prompts"]

            latents = self._generate_latents(
                [prompt],
                num_output_frames=num_output_frames,
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

            self.pipeline.clear_kv_cache()
            if hasattr(self.model, "vae") and hasattr(self.model.vae, "model") and hasattr(self.model.vae.model, "clear_cache"):
                self.model.vae.model.clear_cache()

    def train(self):

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Save the model
            if (not self.config.no_save) and (self.step % self.config.log_iters == 0) and self.step > 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            # Visualization
            if not self.config.no_visualize and (self.step % self.config.log_iters == 0) and self.step > 0:
                if self.is_main_process:
                    print(f"[Visualization] step {self.step}: self.eval_frames {self.eval_frames}, without init latent.")
                was_train_gen = self.model.generator.training
                was_train_txt = self.model.text_encoder.training
                self.model.generator.eval()
                self.model.text_encoder.eval()

                self.parallel_visualization_without_init(
                    rank0_index=0,
                    num_output_frames=self.eval_frames,
                )

                if was_train_gen:
                    self.model.generator.train()
                if was_train_txt:
                    self.model.text_encoder.train()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                if TRAIN_GENERATOR:
                    print(f"step {self.step} in TRAIN_GENERATOR {TRAIN_GENERATOR}, per iteration time {current_time - self.previous_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}")
                else:
                    print(f"step {self.step} in TRAIN_GENERATOR {TRAIN_GENERATOR}, per iteration time {current_time - self.previous_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                if not self.disable_wandb:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                self.previous_time = current_time

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                if self.is_main_process:
                    print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
