"""
stitcher.py — 两图自动拼接 Pipeline
======================================
将 features + homography + ransac 串联为完整的拼接流程。

原 MATLAB 对应：
  Image_stitching_RANSAC.m, my_imfuse.m

改进：
  - 自动计算输出画布大小和偏移量
  - 支持三种融合模式：线性混合、仅图1、仅图2
  - 为 Phase 3 的多频段融合预留接口
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal
from .features import extract_sift, match_features, SIFTResult, MatchResult
from .ransac import find_homography_ransac
from .blending import multi_band_blend
from .exposure import compensate_exposure


BlendMode = Literal['linear', 'multiband', 'first', 'second']


def _compute_corners(H: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    计算图像四个角经过 H 变换后的坐标。

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
    w, h : int
        图像宽、高

    Returns
    -------
    corners : np.ndarray, shape (4, 2)
        变换后四角坐标 (x, y)
    """
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]).T  # (3, 4)

    warped = H @ corners          # (3, 4)
    warped = warped / warped[2]   # 齐次 → 非齐次
    return warped[:2].T           # (4, 2)


def _linear_blend(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    线性混合两张对齐后的图像。

    原 MATLAB my_imfuse.m 做法：
      逐像素判断 → 都有值则 50/50 平均 → 单边有值则直接取

    Parameters
    ----------
    img1, img2 : np.ndarray
        同尺寸的 RGBA 图像（Alpha 通道标记有效区域）

    Returns
    -------
    blended : np.ndarray (RGB)
    """
    h, w = img1.shape[:2]

    # 分离 RGB 和 Alpha
    rgb1, a1 = img1[:, :, :3], img1[:, :, 3:]
    rgb2, a2 = img2[:, :, :3], img2[:, :, 3:]

    # 二值 mask
    mask1 = (a1 > 0).astype(np.float32)
    mask2 = (a2 > 0).astype(np.float32)

    # 重叠区域：50/50
    overlap = mask1 * mask2
    only1 = mask1 * (1 - mask2)
    only2 = mask2 * (1 - mask1)

    blended = (
        overlap * (0.5 * rgb1 + 0.5 * rgb2) +
        only1 * rgb1 +
        only2 * rgb2
    ).astype(np.uint8)

    return blended


def stitch_two_images(
    img1: np.ndarray,
    img2: np.ndarray,
    ransac_threshold: float = 5.0,
    ratio_thresh: float = 0.75,
    nfeatures: int = 0,
    blend_mode: BlendMode = 'multiband',
    exposure_compensation: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    将两张图自动拼接为全景图。

    Pipeline:
      1. SIFT 特征提取
      2. FLANN + ratio test 匹配
      3. RANSAC 估计最优 H
      4. 透视变换 + 平移对齐
      5. 融合

    Parameters
    ----------
    img1 : np.ndarray
        参考图像（保持不变）
    img2 : np.ndarray
        待变换图像（将被 warp 到 img1 坐标系）
    ransac_threshold : float
        RANSAC inlier 阈值（像素）
    ratio_thresh : float
        Lowe's ratio test 阈值
    nfeatures : int
        特征数上限，0 为不限制
    blend_mode : str
        'multiband' — 多频段融合（默认，消除拼接缝）
        'linear'    — 线性 50/50 混合（原 MATLAB 行为）
        'first'     — 仅保留 img1 区域
        'second'    — 仅保留 img2 区域
    exposure_compensation : bool
        是否对 img2 做曝光补偿（默认开启）
    verbose : bool
        是否打印诊断信息

    Returns
    -------
    result : np.ndarray (BGR)
        拼接结果
    info : dict
        诊断信息
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # ---------- Step 1: 特征提取 ----------
    sift1 = extract_sift(img1, nfeatures=nfeatures)
    sift2 = extract_sift(img2, nfeatures=nfeatures)

    if sift1.descriptors is None or sift2.descriptors is None:
        raise RuntimeError("SIFT extraction failed on one or both images")

    # ---------- Step 2: 特征匹配 ----------
    match = match_features(sift1.descriptors, sift2.descriptors,
                           ratio_thresh=ratio_thresh)

    src_pts = np.float32([sift2.keypoints[m.trainIdx].pt for m in match.good_matches])
    dst_pts = np.float32([sift1.keypoints[m.queryIdx].pt for m in match.good_matches])

    if len(src_pts) < 4:
        raise RuntimeError(f"Too few matches: {len(src_pts)} (need ≥4)")

    # ---------- Step 3: RANSAC ----------
    H, inliers, ransac_info = find_homography_ransac(
        src_pts, dst_pts, threshold=ransac_threshold, normalize=True
    )

    if verbose:
        print(f"  Matches: {len(src_pts)} total, "
              f"{ransac_info['n_inliers']} inliers "
              f"({ransac_info['inlier_ratio']:.1%})")
        print(f"  RANSAC: {ransac_info['n_iters']} iters, "
              f"mean error={ransac_info['mean_error']:.2f}px")

    # ---------- Step 4: 透视变换 ----------
    # 转为 BGRA 再 warp，这样变换后透明区域有 alpha=0
    img2_rgba = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    warped = cv2.warpPerspective(
        img2_rgba, H,
        (w1 + w2, h1 + h2),  # 大画布，确保装得下
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)  # 透明背景 (BGRA)
    )

    # 创建带 alpha 通道的 img1（和 warped 同尺寸）
    img1_rgba = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img1_padded = np.zeros_like(warped)
    img1_padded[:h1, :w1] = img1_rgba

    # ---------- Step 5: 曝光补偿 ----------
    if exposure_compensation:
        img1_padded, warped = compensate_exposure(img1_padded, warped)
        if verbose:
            print(f"  Exposure: compensated")

    # ---------- Step 6: 融合 ----------
    if blend_mode == 'multiband':
        blended = multi_band_blend(img1_padded, warped)
    elif blend_mode == 'linear':
        blended = _linear_blend(img1_padded, warped)
    elif blend_mode == 'first':
        blended = img1_padded[:, :, :3]
    else:
        blended = warped[:, :, :3]

    # 裁剪
    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 找非零区域边界
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        result = blended[y:y+h, x:x+w]
    else:
        result = blended

    # ---------- 诊断信息 ----------
    info = {
        'H': H,
        'n_matches': len(src_pts),
        'src_pts': src_pts,
        'dst_pts': dst_pts,
        'inliers': inliers,
        **ransac_info,
        'sift1': sift1,
        'sift2': sift2,
        'match': match,
    }

    return result, info
