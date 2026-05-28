"""
homography.py — 单应性矩阵计算
===============================
DLT (Direct Linear Transform) + SVD，与原 MATLAB get_Homography.m 数学框架一致。

原 MATLAB 对应：
  get_Homography.m, Homography.m

改进：
  - 可选归一化 DLT（Hartley normalization），提升数值稳定性
  - 返回 H 矩阵同时附带重投影误差，方便调试
  - 支持 cv2.warpPerspective 直接使用（无需 projective2d 封装）
"""

import numpy as np
from typing import Tuple, Optional


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray,
                       normalize: bool = True) -> np.ndarray:
    """
    用 DLT (Direct Linear Transform) 计算单应性矩阵。

    对 n 对对应点 (x₁,y₁) → (x₂,y₂)，满足 x₂ = H·x₁（齐次坐标）。
    每对点贡献两个方程，堆叠为 A·h = 0，用 SVD 求解。

    原 MATLAB get_Homography.m:
        A = [...];           % 逐对构建 Ah=0 的系数矩阵
        [~,~,V] = svd(A);    % SVD 分解
        H = V(:,end);         % 最小奇异值对应解
        H = reshape(H,[3,3]);

    这里的数学完全一致，但加入了可选的坐标归一化来提升精度。

    Parameters
    ----------
    src_pts : np.ndarray, shape (N, 2)
        源图像中的点坐标
    dst_pts : np.ndarray, shape (N, 2)
        目标图像中的点坐标
    normalize : bool
        是否做 Hartley 归一化（推荐开启，数值更稳）

    Returns
    -------
    H : np.ndarray, shape (3, 3)
        单应性矩阵，使得 dst_pt_h = H @ src_pt_h
    """
    assert len(src_pts) >= 4, "Need at least 4 point pairs"
    assert src_pts.shape == dst_pts.shape

    n = len(src_pts)
    x1, y1 = src_pts[:, 0], src_pts[:, 1]
    x2, y2 = dst_pts[:, 0], dst_pts[:, 1]

    # ---------------------------------------------------------------
    # Hartley 归一化：将点坐标缩放和平移到均值为 0、平均距离 √2
    # 这能显著提升 DLT 在有限精度下的数值稳定性
    # ---------------------------------------------------------------
    if normalize:
        # 源点归一化
        mean1 = np.mean(src_pts, axis=0)
        std1 = np.mean(np.sqrt((x1 - mean1[0])**2 + (y1 - mean1[1])**2))
        s1 = np.sqrt(2) / std1
        T1 = np.array([[s1, 0, -s1 * mean1[0]],
                       [0, s1, -s1 * mean1[1]],
                       [0, 0, 1]])

        # 目标点归一化
        mean2 = np.mean(dst_pts, axis=0)
        std2 = np.mean(np.sqrt((x2 - mean2[0])**2 + (y2 - mean2[1])**2))
        s2 = np.sqrt(2) / std2
        T2 = np.array([[s2, 0, -s2 * mean2[0]],
                       [0, s2, -s2 * mean2[1]],
                       [0, 0, 1]])

        # 归一化坐标
        src_norm = (T1 @ np.vstack([x1, y1, np.ones(n)])).T[:, :2]
        dst_norm = (T2 @ np.vstack([x2, y2, np.ones(n)])).T[:, :2]
        x1, y1 = src_norm[:, 0], src_norm[:, 1]
        x2, y2 = dst_norm[:, 0], dst_norm[:, 1]

    # ---------------------------------------------------------------
    # 构建系数矩阵 A，每对点两行
    #
    # [x₁  y₁  1   0   0   0  -x₁x₂  -y₁x₂  -x₂]
    # [ 0   0   0  x₁  y₁  1  -x₁y₂  -y₁y₂  -y₂]
    # ---------------------------------------------------------------
    A = np.zeros((2 * n, 9))
    for i in range(n):
        A[2 * i]     = [x1[i], y1[i], 1, 0, 0, 0,
                         -x1[i] * x2[i], -y1[i] * x2[i], -x2[i]]
        A[2 * i + 1] = [0, 0, 0, x1[i], y1[i], 1,
                         -x1[i] * y2[i], -y1[i] * y2[i], -y2[i]]

    # SVD 分解，最小奇异值对应的右奇异向量即解
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]           # Vt 最后一行 = V 最后一列
    H_norm = h.reshape(3, 3)

    # ---------------------------------------------------------------
    # 反归一化
    # ---------------------------------------------------------------
    if normalize:
        H_norm = np.linalg.inv(T2) @ H_norm @ T1

    # 归一化使得 H[2,2] = 1
    if abs(H_norm[2, 2]) > 1e-8:
        H_norm = H_norm / H_norm[2, 2]

    return H_norm


def reprojection_error(H: np.ndarray, src_pts: np.ndarray,
                       dst_pts: np.ndarray) -> np.ndarray:
    """
    计算每对点的重投影误差（欧氏距离）。

    Parameters
    ----------
    H : np.ndarray, shape (3, 3)
    src_pts, dst_pts : np.ndarray, shape (N, 2)

    Returns
    -------
    errors : np.ndarray, shape (N,)
        每对点的欧氏距离误差
    """
    n = len(src_pts)
    src_h = np.hstack([src_pts, np.ones((n, 1))])          # (N, 3)
    proj = (H @ src_h.T).T                                 # (N, 3)
    proj = proj[:, :2] / proj[:, 2:3]                      # 齐次 → 非齐次
    errors = np.sqrt(np.sum((proj - dst_pts) ** 2, axis=1))
    return errors


def mean_reprojection_error(H: np.ndarray, src_pts: np.ndarray,
                            dst_pts: np.ndarray) -> float:
    """平均重投影误差，用于评估 H 矩阵质量。"""
    return float(np.mean(reprojection_error(H, src_pts, dst_pts)))
