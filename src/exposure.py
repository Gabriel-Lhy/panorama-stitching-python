"""
exposure.py — 自动曝光补偿
============================
解决拼接时两图因自动曝光差异导致的亮度不一致。

原理：
  在重叠区域计算两图各通道的平均亮度比，
  用这个比值对较暗的图像做增益补偿。
  本质上是一个全局线性变换，不改动局部对比度。

参考：
  Brown & Lowe (2007) "Automatic Panoramic Image Stitching using Invariant Features"
"""

import numpy as np
from typing import Tuple


def compute_gain(img1_rgba: np.ndarray, img2_rgba: np.ndarray) -> np.ndarray:
    """
    计算曝光补偿的增益系数。

    在重叠区域内：
      gain[c] = mean(I1_overlap[c]) / mean(I2_overlap[c])
    如果分母接近 0，gain[c] = 1.0（不补偿）。

    Parameters
    ----------
    img1_rgba : np.ndarray, shape (H, W, 4)
        参考图 (BGRA)
    img2_rgba : np.ndarray, shape (H, W, 4)
        待补偿图 (BGRA)

    Returns
    -------
    gain : np.ndarray, shape (3,)
        每个通道的增益系数
    """
    # 分离通道
    rgb1 = img1_rgba[:, :, :3].astype(np.float64)
    rgb2 = img2_rgba[:, :, :3].astype(np.float64)
    a1 = img1_rgba[:, :, 3]
    a2 = img2_rgba[:, :, 3]

    # 重叠区域 mask
    overlap = (a1 > 0) & (a2 > 0)

    if overlap.sum() < 100:
        # 重叠区太小，不补偿
        return np.ones(3, dtype=np.float64)

    gain = np.ones(3, dtype=np.float64)
    for c in range(3):
        v1 = rgb1[:, :, c][overlap]
        v2 = rgb2[:, :, c][overlap]

        mean1 = np.mean(v1)
        mean2 = np.mean(v2)

        if mean2 > 1.0:
            gain[c] = mean1 / mean2
        else:
            gain[c] = 1.0

    # 限制增益范围，防止极端补偿
    gain = np.clip(gain, 0.5, 2.0)

    return gain


def apply_gain(img_rgba: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """
    对图像各通道分别乘以增益系数。

    Parameters
    ----------
    img_rgba : np.ndarray, shape (H, W, 4)
    gain : np.ndarray, shape (3,)

    Returns
    -------
    compensated : np.ndarray, shape (H, W, 4)
    """
    compensated = img_rgba.copy().astype(np.float64)
    for c in range(3):
        mask = compensated[:, :, 3] > 0
        compensated[:, :, c][mask] *= gain[c]
    return np.clip(compensated, 0, 255).astype(np.uint8)


def compensate_exposure(img1_rgba: np.ndarray,
                        img2_rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    一键曝光补偿：以 img1 为基准，调整 img2。

    Parameters
    ----------
    img1_rgba, img2_rgba : np.ndarray, shape (H, W, 4)
        两张对齐后的 BGRA 图像

    Returns
    -------
    (img1_rgba, img2_compensated) : Tuple[np.ndarray, np.ndarray]
        img1 不动，img2 经过增益调整
    """
    gain = compute_gain(img1_rgba, img2_rgba)
    img2_comp = apply_gain(img2_rgba, gain)
    return img1_rgba, img2_comp
