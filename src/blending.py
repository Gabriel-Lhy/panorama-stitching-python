"""
blending.py — 多频段融合 (Multi-band Blending)
================================================
用拉普拉斯金字塔在不同频率层分别融合，消除简单线性混合的拼接缝。

原 MATLAB 对应：
  my_imfuse.m（仅做了 50/50 逐像素平均）

原理：
  将两图分解为不同频率的带通层（拉普拉斯金字塔），
  在每一层用高斯金字塔加权的 mask 做过渡，
  最后从顶层重建。低频（大尺度）过渡宽，高频（细节）过渡窄，
  这样拼接缝在人眼中不可见。

参考：
  Burt & Adelson (1983) "A Multiresolution Spline With Application to Image Mosaics"
"""

import cv2
import numpy as np
from typing import Tuple, List


def _gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    构建高斯金字塔。

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, C)
    levels : int
        金字塔层数

    Returns
    -------
    pyramid : list of np.ndarray
        pyramid[0] = 原图, pyramid[i] = 第 i 次下采样结果
    """
    pyramid = [image.astype(np.float32)]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image.astype(np.float32))
    return pyramid


def _laplacian_pyramid(gaussian_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    从高斯金字塔构建拉普拉斯金字塔。

    L_i = G_i - expand(G_{i+1})
    顶层 L_{n-1} = G_{n-1}（最高层直接保留）

    Parameters
    ----------
    gaussian_pyr : list of np.ndarray

    Returns
    -------
    laplacian_pyr : list of np.ndarray
    """
    laplacian_pyr = []
    n = len(gaussian_pyr)

    for i in range(n - 1):
        # 上采样 G_{i+1} 到 G_i 的尺寸
        h, w = gaussian_pyr[i].shape[:2]
        expanded = cv2.pyrUp(gaussian_pyr[i + 1],
                             dstsize=(w, h))
        # L_i = G_i - expand(G_{i+1})
        lap = gaussian_pyr[i] - expanded
        laplacian_pyr.append(lap)

    # 顶层 = 最高层高斯
    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr


def _blend_pyramids(lap1: List[np.ndarray], lap2: List[np.ndarray],
                    mask_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    在每一层用 mask 混合两个拉普拉斯金字塔。

    result_i = mask_i * lap1_i + (1 - mask_i) * lap2_i
    """
    blended = []
    for l1, l2, m in zip(lap1, lap2, mask_pyr):
        # mask 可能需要扩展到和图像相同的通道数
        if m.ndim == 2 and l1.ndim == 3:
            m = np.repeat(m[:, :, np.newaxis], l1.shape[2], axis=2)
        b = m * l1 + (1.0 - m) * l2
        blended.append(b)
    return blended


def _reconstruct_from_laplacian(laplacian_pyr: List[np.ndarray]) -> np.ndarray:
    """
    从混合后的拉普拉斯金字塔重建图像。

    从顶层开始，逐层 expand 并加上当前层的拉普拉斯。
    """
    result = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        h, w = laplacian_pyr[i].shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h))
        result += laplacian_pyr[i]
    return result


def _compute_levels(h: int, w: int) -> int:
    """根据图像尺寸自动计算金字塔层数。最小层尺寸 ≥ 32 像素。"""
    return int(np.log2(min(h, w))) - 4  # 顶层约 16~32px


def multi_band_blend(img1_rgba: np.ndarray, img2_rgba: np.ndarray,
                     levels: int = None, border_width: int = 50) -> np.ndarray:
    """
    用拉普拉斯金字塔多频段融合两张已对齐的 RGBA 图像。

    对比原 my_imfuse.m 的 50/50 平均：
      - 低频层（大尺度光照变化）：mask 过渡带宽，平滑过渡
      - 高频层（纹理细节）：mask 过渡带窄，保持细节清晰
      → 结果：接缝消隐，纹理保留

    Parameters
    ----------
    img1_rgba : np.ndarray, shape (H, W, 4)
        参考图 (BGRA)
    img2_rgba : np.ndarray, shape (H, W, 4)
        变换后的图 (BGRA)
    levels : int or None
        金字塔层数，None 则自动计算
    border_width : int
        重叠区边缘的过渡带宽（像素），越大过渡越平滑

    Returns
    -------
    result : np.ndarray, shape (H, W, 3)
        融合后的 BGR 图像
    """
    h, w = img1_rgba.shape[:2]

    # 分离 RGB 和 Alpha
    rgb1 = img1_rgba[:, :, :3].astype(np.float32)
    rgb2 = img2_rgba[:, :, :3].astype(np.float32)
    a1 = img1_rgba[:, :, 3].astype(np.float32)
    a2 = img2_rgba[:, :, 3].astype(np.float32)

    # 二值有效区域
    mask1 = (a1 > 0).astype(np.float32)
    mask2 = (a2 > 0).astype(np.float32)

    # 只取重叠区域 + 边缘扩展（减少计算量）
    overlap = mask1 * mask2
    if overlap.sum() < 100:
        # 几乎没有重叠，直接拼接
        result = np.zeros((h, w, 3), dtype=np.float32)
        result[mask1 > 0] = rgb1[mask1 > 0]
        result[mask2 > 0] = rgb2[mask2 > 0]
        return np.clip(result, 0, 255).astype(np.uint8)

    # 确定金字塔层数
    if levels is None:
        levels = _compute_levels(h, w)
    levels = max(2, min(levels, _compute_levels(h, w)))

    # ---------------------------------------------------------------
    # 构建混合权重 mask
    #
    #   img1 侧: 逐步衰减到 0
    #   重叠区: 从 1 线性过渡到 0
    #   img2 侧: 保持 0
    #
    #   mask=1 表示取 img1，mask=0 表示取 img2
    # ---------------------------------------------------------------
    overlap_region = overlap.astype(np.uint8)

    # 找到重叠区域的水平过渡方向
    ys, xs = np.where(overlap_region > 0)
    if len(xs) == 0:
        result = np.zeros((h, w, 3), dtype=np.float32)
        result[mask1 > 0] = rgb1[mask1 > 0]
        result[mask2 > 0] = rgb2[mask2 > 0]
        return np.clip(result, 0, 255).astype(np.uint8)

    x_min, x_max = xs.min(), xs.max()

    # 创建过渡权重：在重叠区域内从 1→0
    blend_mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        row_mask = overlap_region[y]
        if row_mask.sum() == 0:
            continue
        col_indices = np.where(row_mask > 0)[0]
        c_min, c_max = col_indices[0], col_indices[-1]

        if c_max - c_min < 2:
            blend_mask[y, c_min:c_max+1] = 0.5
        else:
            # 线性过渡：左→右，1→0
            ramp = np.linspace(1.0, 0.0, c_max - c_min + 1)
            blend_mask[y, c_min:c_max+1] = ramp

    # img1 独有区域保持 mask=1
    only1 = mask1 * (1 - mask2)
    blend_mask[only1 > 0] = 1.0

    # ---------------------------------------------------------------
    # 构建高斯金字塔用于 mask（平滑过渡边界）
    # ---------------------------------------------------------------
    mask_pyr = _gaussian_pyramid(blend_mask, levels)

    # 构建拉普拉斯金字塔
    lap1 = _laplacian_pyramid(_gaussian_pyramid(rgb1, levels))
    lap2 = _laplacian_pyramid(_gaussian_pyramid(rgb2, levels))

    # 逐层混合
    blended_pyr = _blend_pyramids(lap1, lap2, mask_pyr)

    # 重建
    result = _reconstruct_from_laplacian(blended_pyr)

    return np.clip(result, 0, 255).astype(np.uint8)
