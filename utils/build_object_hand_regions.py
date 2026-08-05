import torch
import torch.nn.functional as F

def build_object_hand_regions(
    mask_video,
    pose_video,
    clean_latent,
    hand_expand_pixels=10,
):
    """
    输入:
        mask_video:   [B, 81, 3, H0, W0], uint8 or float
        pose_video:   [B, 81, 3, H0, W0], uint8 or float
        clean_latent: [B, 21, C, h, w]

    输出:
        object_region_21: [B, 21, h, w]
        hand_region_21:   [B, 21, h, w]
    """

    # 1. 提取二值区域
    object_region = (mask_video.sum(dim=2) > 0).float()   # [B,81,H0,W0]
    hand_region = (pose_video.sum(dim=2) > 0).float()     # [B,81,H0,W0]

    B, T, H0, W0 = object_region.shape
    _, T_latent, _, h, w = clean_latent.shape

    # 2. 对 hand_region 做膨胀，扩充上下左右各 hand_expand_pixels
    if hand_expand_pixels > 0:
        k = 2 * hand_expand_pixels + 1
        hand_region = F.max_pool2d(
            hand_region.view(B * T, 1, H0, W0),
            kernel_size=k,
            stride=1,
            padding=hand_expand_pixels
        ).view(B, T, H0, W0)

    # 3. 空间下采样到 latent 尺寸
    object_region = F.interpolate(
        object_region.view(B * T, 1, H0, W0),
        size=(h, w),
        mode="nearest"
    ).view(B, T, h, w)

    hand_region = F.interpolate(
        hand_region.view(B * T, 1, H0, W0),
        size=(h, w),
        mode="nearest"
    ).view(B, T, h, w)

    # 4. 时间压缩：81 -> 21
    keep_idx = [0] + [1 + 4 * i for i in range((T - 1) // 4)]

    object_region_21 = object_region[:, keep_idx, :, :]
    hand_region_21 = hand_region[:, keep_idx, :, :]

    # 5. 显式二值化（虽然 nearest 下通常已经是 0/1）
    object_region_21 = (object_region_21 > 0.5).float()
    hand_region_21 = (hand_region_21 > 0.5).float()

    # 6. 检查
    assert object_region_21.shape[1] == T_latent, \
        f"object_region_21 frames={object_region_21.shape[1]}, but clean_latent frames={T_latent}"
    assert hand_region_21.shape[1] == T_latent, \
        f"hand_region_21 frames={hand_region_21.shape[1]}, but clean_latent frames={T_latent}"

    return object_region_21, hand_region_21