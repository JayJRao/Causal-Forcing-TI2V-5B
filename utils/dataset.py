from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os

import pandas as pd
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from torchvision import transforms


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, min_num_frames=None, max_num_frames=None, frame_interval=1, num_frames=81,
                 height=480, width=832, is_i2v=False,
                 video_path_col='video_path', text_col='caption'):
        """
        TextVideoDataset for loading videos from CSV

        Args:
            csv_path: Path to CSV file containing video metadata
            min_num_frames: Minimum number of frames required in video (default: None, no limit)
            max_num_frames: Maximum number of frames allowed in video (default: None, no limit)
            frame_interval: Interval between sampled frames
            num_frames: Number of frames to sample
            height: Target height for frames
            width: Target width for frames
            is_i2v: Whether this is image-to-video mode
            video_path_col: Column name in CSV containing video file paths (default: 'video_path')
            text_col: Column name in CSV containing text captions (default: 'caption')
        """
        metadata = pd.read_csv(csv_path)

        self.path = metadata[video_path_col].to_list()
        self.prompt = metadata[text_col].to_list()

        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"[TextVideoDataset] csv_path={csv_path}, num_samples={len(self.path)}, "
              f"num_frames={num_frames}, height={height}, width={width}, "
              f"frame_interval={frame_interval}, min_num_frames={min_num_frames}, "
              f"max_num_frames={max_num_frames}, is_i2v={is_i2v}, "
              f"video_path_col={video_path_col}, text_col={text_col}")

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, min_num_frames, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()

        # Check frame count limits
        if min_num_frames is not None and total_frames < min_num_frames:
            reader.close()
            return None

        if max_num_frames is not None and total_frames > max_num_frames:
            reader.close()
            return None

        # Check if we have enough frames for sampling
        if total_frames - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.min_num_frames, self.max_num_frames,
                                                start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        while True:
            try:
                prompt = self.prompt[data_id]
                path = self.path[data_id]
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"prompts": prompt, "frames": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"prompts": prompt, "frames": video, "path": path}
                break
            except:
                data_id += 1
        return data

    def __len__(self):
        return len(self.path)


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



class I2VDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        num_frames=81,
        sample_mode="uniform",
        return_video=True,
        return_image=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.return_video = return_video
        self.return_image = return_image

        self.center_crop = center_crop
        self.random_flip = random_flip

        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        data_id = (index) % len(self.video_paths)

        video_path = self.video_paths[int(data_id)]
        if video_path.startswith("/m2v_intern/xuziyi06/"):
            video_path = video_path.replace(
                "/m2v_intern/xuziyi06/",
                "/m2v_intern/raozejing/xuziyi06/"
            )
        if video_path.startswith("/share/xuziyi/"):
            video_path = video_path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )

        text = self.texts[int(data_id)]

        out = {"prompts": text, "idx": index, "data_id": int(data_id), "video_path": video_path}

        frames_pil = None
        if self.return_video:
            frames_pil = self._read_video_frames(video_path, num_frames=self.num_frames, mode=self.sample_mode)

        if self.return_image:
            if frames_pil is not None and len(frames_pil) > 0:
                image = frames_pil[0]
            else:
                image = self._read_first_frame(video_path)
            out["images"] = self._process_single_image(image)

        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(frames_pil)

        return out

    def _cover_resize_pil(self, image: Image.Image) -> Image.Image:
        target_height, target_width = self.height, self.width
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        shape = [round(h * scale), round(w * scale)]
        image = torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )
        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        image = self._cover_resize_pil(image)
        image = self.post_processor(image)
        return image

    def _process_video_frames_consistent(self, frames_pil):
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty video frames")

        frames_resized = [self._cover_resize_pil(im) for im in frames_pil]

        if self.center_crop:
            crop_params = None
        else:
            i, j, h, w = transforms.RandomCrop.get_params(
                frames_resized[0], output_size=(self.height, self.width)
            )
            crop_params = (i, j, h, w)

        do_flip = False
        if self.random_flip:
            do_flip = bool(torch.rand(1).item() < 0.5)

        video_t = []
        for im in frames_resized:
            if crop_params is None:
                im2 = torchvision.transforms.functional.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im2 = torchvision.transforms.functional.crop(im, i, j, h, w)

            if do_flip:
                im2 = torchvision.transforms.functional.hflip(im2)

            t = torchvision.transforms.functional.to_tensor(im2)
            t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
            video_t.append(t)

        video = torch.stack(video_t, dim=0)
        return video

    @staticmethod
    def _read_first_frame(video_path: str) -> Image.Image:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()
            return Image.fromarray(frame0).convert("RGB")
        except Exception:
            pass

        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _read_video_frames(video_path: str, num_frames: int, mode: str = "uniform"):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            if vlen <= 0:
                raise RuntimeError("empty video")

            if mode == "head":
                idx = list(range(min(num_frames, vlen)))
            else:
                if vlen >= num_frames:
                    idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
                else:
                    idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

            frames = vr.get_batch(idx).asnumpy()
            return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]
        except Exception:
            pass

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen <= 0:
            cap.release()
            raise RuntimeError(f"empty video: {video_path}")

        if mode == "head":
            idx = list(range(min(num_frames, vlen)))
            if vlen < num_frames:
                idx += [vlen - 1] * (num_frames - vlen)
        else:
            if vlen >= num_frames:
                idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
            else:
                idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class I2VDataset_with_mask_pose(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        mask_col="mask",
        pose_video_col="pose_video",
        num_frames=81,
        sample_mode="uniform",         # "uniform" or "head"
        return_video=True,
        return_image=True,
        return_mask_video=True,
        return_pose_video=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()

        self.mask_paths = self.df[mask_col].tolist() if mask_col in self.df.columns else None
        self.pose_video_paths = self.df[pose_video_col].tolist() if pose_video_col in self.df.columns else None

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_mode = sample_mode

        self.return_video = return_video
        self.return_image = return_image
        self.return_mask_video = return_mask_video
        self.return_pose_video = return_pose_video

        self.center_crop = center_crop
        self.random_flip = random_flip

        # 仅用于 image 分支；video 分支我们手动保证整段视频共享同一组 crop/flip
        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # 保持和你原来一致的“可扰动遍历”写法
        data_id = torch.randint(0, len(self.video_paths), (1,))[0]
        data_id = (data_id + index) % len(self.video_paths)
        data_id = int(data_id)

        video_path = self._remap_path(self.video_paths[data_id])
        text = self.texts[data_id]

        out = {
            "prompts": text,
            "idx": index,
            "data_id": data_id,
            "video_path": video_path,
        }

        mask_path = None
        pose_video_path = None

        if self.mask_paths is not None and self.return_mask_video:
            mask_path = self._remap_path(self.mask_paths[data_id])
            out["mask_video_path"] = mask_path

        if self.pose_video_paths is not None and self.return_pose_video:
            pose_video_path = self._remap_path(self.pose_video_paths[data_id])
            out["pose_video_path"] = pose_video_path

        # 先在主视频上确定采样帧索引，mask / pose_video 共享这组 idx
        frame_indices = None
        frames_pil = None
        if self.return_video or self.return_image or self.return_mask_video or self.return_pose_video:
            frames_pil, frame_indices = self._read_video_frames_with_indices(
                video_path,
                num_frames=self.num_frames,
                mode=self.sample_mode
            )

        # image：直接取主视频首帧
        if self.return_image:
            if frames_pil is not None and len(frames_pil) > 0:
                image = frames_pil[0]
            else:
                image = self._read_first_frame(video_path)
            out["images"] = self._process_single_image(image)

        # 生成统一的空间增强参数，供 frames / mask_video / pose_video 共用
        if frames_pil is not None:
            transform_params = self._get_video_transform_params(frames_pil[0])
        else:
            transform_params = None

        # 主视频：float32, [-1,1]
        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(
                frames_pil,
                transform_params=transform_params
            )

        # mask 视频：uint8, [0,255]，但空间变换链条与 frames 完全一致
        if self.return_mask_video and mask_path is not None:
            mask_frames_pil = self._read_video_frames_by_indices(mask_path, frame_indices)
            out["mask_video"] = self._process_aux_video_frames_consistent(
                mask_frames_pil,
                transform_params=transform_params,
                interpolation=InterpolationMode.BILINEAR,
            )

        # pose 视频：uint8, [0,255]，但空间变换链条与 frames 完全一致
        if self.return_pose_video and pose_video_path is not None:
            pose_frames_pil = self._read_video_frames_by_indices(pose_video_path, frame_indices)
            out["pose_video"] = self._process_aux_video_frames_consistent(
                pose_frames_pil,
                transform_params=transform_params,
                interpolation=InterpolationMode.BILINEAR,
            )

        return out

    # ---------- processing helpers ----------

    def _cover_resize_pil(self, image: Image.Image, interpolation=InterpolationMode.BILINEAR) -> Image.Image:
        """
        先 resize 到能覆盖目标 H/W，再 crop。
        """
        target_height, target_width = self.height, self.width
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        shape = [round(h * scale), round(w * scale)]  # [H, W]
        image = TF.resize(image, shape, interpolation=interpolation)
        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        """
        给首帧 image 用：输出 [3,H,W], float32, [-1,1]
        """
        image = self._cover_resize_pil(image, interpolation=InterpolationMode.BILINEAR)
        image = self.post_processor(image)
        return image

    def _get_video_transform_params(self, ref_image: Image.Image):
        """
        为整段视频生成一组固定的空间增强参数，供 frames / mask_video / pose_video 共用。
        关键是保证三路视频的空间变换完全一致。
        """
        ref_image = self._cover_resize_pil(ref_image, interpolation=InterpolationMode.BILINEAR)

        if self.center_crop:
            crop_params = None
        else:
            i, j, h, w = transforms.RandomCrop.get_params(
                ref_image, output_size=(self.height, self.width)
            )
            crop_params = (i, j, h, w)

        do_flip = False
        if self.random_flip:
            do_flip = bool(torch.rand(1).item() < 0.5)

        return {
            "crop_params": crop_params,
            "do_flip": do_flip,
        }

    def _process_video_frames_consistent(self, frames_pil, transform_params):
        """
        主视频 frames:
        - resize/crop/flip 与 mask/pose 完全一致
        - 输出 float32, [-1,1]
        """
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty video frames")

        crop_params = transform_params["crop_params"]
        do_flip = transform_params["do_flip"]

        video_t = []
        for im in frames_pil:
            im = self._cover_resize_pil(im, interpolation=InterpolationMode.BILINEAR)

            if crop_params is None:
                im = TF.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im = TF.crop(im, i, j, h, w)

            if do_flip:
                im = TF.hflip(im)

            t = TF.to_tensor(im)               # float32, [0,1]
            t = TF.normalize(t, [0.5], [0.5]) # float32, [-1,1]
            video_t.append(t)

        video = torch.stack(video_t, dim=0)   # [T, 3, H, W]
        return video

    def _process_aux_video_frames_consistent(self, frames_pil, transform_params, interpolation=InterpolationMode.BILINEAR):
        """
        给 mask_video / pose_video 用：
        - 与 frames 完全一致的 resize/crop/flip/interpolation
        - 不做数值缩放
        - 返回 uint8, [0,255]
        """
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty auxiliary video frames")

        crop_params = transform_params["crop_params"]
        do_flip = transform_params["do_flip"]

        video_t = []
        for im in frames_pil:
            im = self._cover_resize_pil(im, interpolation=interpolation)

            if crop_params is None:
                im = TF.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im = TF.crop(im, i, j, h, w)

            if do_flip:
                im = TF.hflip(im)

            t = TF.pil_to_tensor(im)           # uint8, [C,H,W], [0,255]
            video_t.append(t)

        video = torch.stack(video_t, dim=0)   # [T, 3, H, W], uint8
        return video

    # ---------- IO helpers ----------

    @staticmethod
    def _remap_path(path: str) -> str:
        if isinstance(path, str) and path.startswith("/m2v_intern/xuziyi06/"):
            path = path.replace(
                "/m2v_intern/xuziyi06/",
                "/m2v_intern/raozejing/xuziyi06/"
            )
        if isinstance(path, str) and path.startswith("/share/xuziyi/"):
            path = path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )
        return path

    @staticmethod
    def _read_first_frame(video_path: str) -> Image.Image:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()  # RGB
            return Image.fromarray(frame0).convert("RGB")
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _build_frame_indices(vlen: int, num_frames: int, mode: str = "uniform"):
        """
        生成统一采样帧索引。
        """
        if vlen <= 0:
            raise RuntimeError("empty video")

        if mode == "head":
            idx = list(range(min(num_frames, vlen)))
            if vlen < num_frames:
                idx += [vlen - 1] * (num_frames - vlen)
        else:
            # uniform
            if vlen >= num_frames:
                idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
            else:
                idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

        return idx

    @classmethod
    def _read_video_frames_with_indices(cls, video_path: str, num_frames: int, mode: str = "uniform"):
        """
        返回:
            frames_pil: List[PIL.Image]
            idx: List[int]
        """
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            idx = cls._build_frame_indices(vlen, num_frames, mode)
            frames = vr.get_batch(idx).asnumpy()  # (T,H,W,3), RGB
            frames_pil = [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]
            return frames_pil, idx
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = cls._build_frame_indices(vlen, num_frames, mode)

        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames, idx

    @staticmethod
    def _read_video_frames_by_indices(video_path: str, idx):
        """
        按主视频已经确定好的 idx 读取另一段视频，保证时间采样严格对齐。
        """
        if idx is None or len(idx) == 0:
            raise RuntimeError("Empty frame index list")

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            safe_idx = [min(max(int(i), 0), vlen - 1) for i in idx]
            frames = vr.get_batch(safe_idx).asnumpy()  # (T,H,W,3), RGB
            return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]
        except Exception:
            pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen <= 0:
            cap.release()
            raise RuntimeError(f"empty video: {video_path}")

        safe_idx = [min(max(int(i), 0), vlen - 1) for i in idx]

        frames = []
        for fi in safe_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames

def cycle(dl):
    while True:
        for data in dl:
            yield data
