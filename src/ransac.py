"""
ransac.py — RANSAC 鲁棒单应性估计
====================================
用 RANSAC 从含噪匹配点中估计最优单应性矩阵。

原 MATLAB 对应：
  RANSAC_find_inliers.m

改进：
  - 自适应迭代次数：根据当前 inlier 比例动态计算所需迭代数，而非固定 3000 次
  - 提前收敛：当找到足够多 inliers（>= 总匹配数的 80%）时立即退出
  - 用 Hartley 归一化 DLT 替代裸 DLT，数值稳定性更好
"""

import numpy as np
from typing import Tuple, Optional
from .homography import compute_homography, reprojection_error


def find_homography_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 5.0,
    max_iters: int = 3000,
    confidence: float = 0.99,
    min_samples: int = 4,
    early_stop_ratio: float = 0.8,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    用 RANSAC 找到最佳单应性矩阵。

    原 MATLAB 做法：
      - 固定迭代 3000 次
      - 每轮随机选 5 对点（n=5）
      - 重投影误差 < 5 像素算 inlier
      - 保留 inlier 最多的 H

    这里的改进：
      1. 自适应迭代次数：N = log(1-p) / log(1-w^k)，其中 w 是 inlier 比例
      2. 提前收敛：inlier 占比 >= early_stop_ratio 时立即返回
      3. 使用 Hartley 归一化 DLT（提升数值稳定）
      4. 返回诊断信息（迭代次数、inlier 比例、平均误差）

    Parameters
    ----------
    src_pts, dst_pts : np.ndarray, shape (N, 2)
        匹配点坐标
    threshold : float
        inlier 判定阈值（像素），默认 5.0
    max_iters : int
        最大迭代次数上限，默认 3000
    confidence : float
        期望置信度，用于自适应计算迭代数
    min_samples : int
        每轮采样点数，4 是 DLT 的理论下限
    early_stop_ratio : float
        inlier 占比超过此值则提前终止
    normalize : bool
        是否使用 Hartley 归一化

    Returns
    -------
    H : np.ndarray, shape (3, 3)
        最优单应性矩阵
    inlier_mask : np.ndarray, shape (N,), dtype=bool
        标记哪些点是 inlier
    info : dict
        诊断信息：{'n_iters', 'n_inliers', 'inlier_ratio', 'mean_error'}
    """
    n_pts = len(src_pts)
    if n_pts < min_samples:
        raise ValueError(f"Need at least {min_samples} point pairs, got {n_pts}")

    best_H = None
    best_inliers = np.array([], dtype=bool)
    best_count = 0

    n_iters = 0
    target_iters = max_iters  # 自适应迭代数，初始为上限

    for n_iters in range(1, max_iters + 1):
        # -------- 随机采样 --------
        indices = np.random.choice(n_pts, size=min_samples, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]

        # -------- 估计 H --------
        try:
            H = compute_homography(sample_src, sample_dst, normalize=normalize)
        except np.linalg.LinAlgError:
            continue  # 退化配置，跳过

        # -------- 计算 inliers --------
        errors = reprojection_error(H, src_pts, dst_pts)
        inliers = errors < threshold
        count = np.sum(inliers)

        # -------- 更新最优 --------
        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

            # 自适应更新迭代次数
            w = best_count / n_pts                     # inlier 比例
            w = max(w, 1e-6)
            k = min_samples
            target_iters = int(
                np.log(1 - confidence) / np.log(1 - w**k)
            )
            target_iters = min(target_iters, max_iters)

        # -------- 提前终止 --------
        if best_count / n_pts >= early_stop_ratio:
            break

        if n_iters >= target_iters:
            break

    # -------- 用所有 inlier 重新精炼 H --------
    if best_H is not None and best_count >= min_samples:
        best_H = compute_homography(
            src_pts[best_inliers], dst_pts[best_inliers],
            normalize=normalize
        )

    info = {
        'n_iters': n_iters,
        'n_inliers': best_count,
        'inlier_ratio': best_count / n_pts if n_pts > 0 else 0,
        'mean_error': float(np.mean(
            reprojection_error(best_H, src_pts[best_inliers], dst_pts[best_inliers])
        )) if best_count > 0 else float('inf'),
    }

    return best_H, best_inliers, info
