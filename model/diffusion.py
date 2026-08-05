from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.wan_wrapper_22 import Wan22_DiffusionWrapper, Wan22_VAEWrapper
from utils.global_config import set_wan_version, get_wan_version


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        # self.num_train_timestep = args.num_train_timestep
        # self.min_step = int(0.02 * self.num_train_timestep)
        # self.max_step = int(0.98 * self.num_train_timestep)
        # self.guidance_scale = args.guidance_scale
        # self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

    def _initialize_models(self, args, device):
        set_wan_version(args, str(self.__class__))
        if get_wan_version(str(self.__class__)) == "2.2":
            _DiffusionWrapper = Wan22_DiffusionWrapper
            _VAEWrapper = Wan22_VAEWrapper
        else:
            _DiffusionWrapper = WanDiffusionWrapper
            _VAEWrapper = WanVAEWrapper

        self.generator = _DiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = _VAEWrapper()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        object_region: torch.Tensor = None,
        hand_region: torch.Tensor = None,
        object_weight: int = 1,
        hand_weight: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)

        if self.independent_first_frame:
            timestep[:, 0] = 0
        print('timestep: ',timestep)

        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        if self.independent_first_frame:
            noisy_latents[:, 0] = clean_latent[:, 0]

        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None
        )

        per_pixel_loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        )

        loss_weight = torch.ones(
            (batch_size, num_frame, 1, flow_pred.shape[-2], flow_pred.shape[-1]),
            device=flow_pred.device,
            dtype=per_pixel_loss.dtype,
        )

        if object_region is not None:
            object_region = object_region.to(device=flow_pred.device, dtype=per_pixel_loss.dtype)
            loss_weight = loss_weight + (object_weight - 1.0) * object_region.unsqueeze(2)

        if hand_region is not None:
            hand_region = hand_region.to(device=flow_pred.device, dtype=per_pixel_loss.dtype)
            loss_weight = loss_weight + (hand_weight - 1.0) * hand_region.unsqueeze(2)

        weighted_loss = per_pixel_loss * loss_weight
        loss = weighted_loss.mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict
