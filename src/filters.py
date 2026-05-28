"""
filters.py — 图像卷积核实现
================================
手写 Gaussian / Sobel / Haar-like 滤波器，替代原 MATLAB 版本。
核心理念：理解卷积的底层机制，同时用 NumPy 向量化替代逐像素循环。

原 MATLAB 项目对应文件：
  conv2d.m, set_scale.m, Gaussian.m, Sobel.m, Haarlike.m

改进点：
  - 用 np.kron 替代 set_scale 的双重循环
  - 用 scipy.signal.convolve2d 提供高效卷积，同时保留手写版本做教学对比
  - 统一的 apply_* 接口，输入 RGB/灰度均可
"""

import numpy as np
from scipy.signal import convolve2d
from typing import Tuple, Optional, Union


# ---------------------------------------------------------------------------
# 底层：卷积 & 核缩放
# ---------------------------------------------------------------------------

def conv2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    手写 2D 卷积（full 模式），与原 MATLAB conv2d.m 逻辑一致。
    教学用途为主；生产代码请用 scipy.signal.convolve2d。

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        输入图像（二维灰度）
    kernel : np.ndarray, shape (h, w)
        卷积核

    Returns
    -------
    result : np.ndarray, shape (H+h-1, W+w-1)
    """
    H, W = image.shape
    h, w = kernel.shape
    result = np.zeros((H + h - 1, W + w - 1))

    for i in range(h):
        for j in range(w):
            result[i:i+H, j:j+W] += kernel[i, j] * image

    return result


def conv2d(image: np.ndarray, kernel: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    高效 2D 卷积，封装 scipy.signal.convolve2d。

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    kernel : np.ndarray, shape (h, w)
    mode : str
        'full' — 原 MATLAB 行为（默认）
        'same' — 输出与输入同尺寸
        'valid' — 仅有效区域

    Returns
    -------
    result : np.ndarray
    """
    return convolve2d(image, kernel, mode=mode)


def set_scale(kernel: np.ndarray, scale: int) -> np.ndarray:
    """
    放大卷积核：把每个元素复制成 scale×scale 的块。

    原 MATLAB set_scale.m 用双重循环逐个赋值；
    这里用 np.kron 一行完成，语义等价但效率高得多。

    Parameters
    ----------
    kernel : np.ndarray, shape (h, w)
        原始小核
    scale : int
        放大倍数

    Returns
    -------
    scaled : np.ndarray, shape (h*scale, w*scale)
    """
    return np.kron(kernel, np.ones((scale, scale)))


# ---------------------------------------------------------------------------
# 卷积核生成器
# ---------------------------------------------------------------------------

def gaussian_kernel(size: int = 3) -> np.ndarray:
    """
    生成高斯平滑核。

    原 MATLAB 硬编码了 3×3 的 [1 2 1; 2 4 2; 1 2 1]/16。
    这里扩展为通用版本，支持任意奇数尺寸。

    Parameters
    ----------
    size : int
        核尺寸（奇数），默认 3

    Returns
    -------
    kernel : np.ndarray, shape (size, size)
        归一化高斯核
    """
    if size == 3:
        # 与原 MATLAB 完全一致的 3×3 高斯核
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float64)
        return kernel / 16.0

    # 通用高斯核：σ = size/6（覆盖 ±3σ 约整个窗口）
    sigma = size / 6.0
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def sobel_kernels(scale: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Sobel 边缘检测核（x 方向和 y 方向）。

    Parameters
    ----------
    scale : int
        核放大倍数，默认 1

    Returns
    -------
    Gx, Gy : (np.ndarray, np.ndarray)
        x 方向梯度核, y 方向梯度核
    """
    sobel_x = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    sobel_y = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)

    if scale > 1:
        sobel_x = set_scale(sobel_x, scale)
        sobel_y = set_scale(sobel_y, scale)

    return sobel_x, sobel_y


def haar_kernel(kernel_type: str = 'Haar12', scale: int = 1) -> np.ndarray:
    """
    生成 Haar-like 特征核。

    Parameters
    ----------
    kernel_type : str
        可选: 'Haar12', 'Haar21', 'Haar13', 'Haar31', 'Haar22'
    scale : int
        核放大倍数，默认 1

    Returns
    -------
    kernel : np.ndarray
    """
    kernels = {
        'Haar12': np.array([[-1, 1]], dtype=np.float64),           # 水平二段
        'Haar21': np.array([[-1], [1]], dtype=np.float64),         # 垂直二段
        'Haar13': np.array([[-1, 1, -1]], dtype=np.float64),       # 水平三段
        'Haar31': np.array([[-1], [1], [-1]], dtype=np.float64),   # 垂直三段
        'Haar22': np.array([[1, -1], [-1, 1]], dtype=np.float64),  # 棋盘格
    }

    if kernel_type not in kernels:
        raise ValueError(f"Unknown Haar kernel: {kernel_type}. "
                         f"Choose from {list(kernels.keys())}")

    kernel = kernels[kernel_type]
    if scale > 1:
        kernel = set_scale(kernel, scale)
    return kernel


# ---------------------------------------------------------------------------
# 高层接口：直接对图像应用滤波器
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """RGB/BGR → 灰度（如已是灰度则直接返回）"""
    if image.ndim == 2:
        return image.astype(np.float64)
    # OpenCV 默认 BGR
    if image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image.astype(np.float64)


def apply_gaussian(image: np.ndarray, kernel_size: int = 3, scale: int = 1) -> np.ndarray:
    """
    对图像应用高斯平滑。

    Parameters
    ----------
    image : np.ndarray
        RGB 或灰度图像
    kernel_size : int
        高斯核尺寸
    scale : int
        核放大倍数

    Returns
    -------
    result : np.ndarray
    """
    gray = _to_grayscale(image)
    kernel = gaussian_kernel(kernel_size)
    if scale > 1:
        kernel = set_scale(kernel, scale)
    return conv2d(gray, kernel, mode='same')


def apply_sobel(image: np.ndarray, scale: int = 1) -> np.ndarray:
    """
    对图像应用 Sobel 边缘检测。

    Parameters
    ----------
    image : np.ndarray
    scale : int

    Returns
    -------
    gradient_magnitude : np.ndarray
        梯度幅值（与原 MATLAB Sobel.m 的输出一致）
    """
    gray = _to_grayscale(image)
    Gx, Gy = sobel_kernels(scale)
    conv_x = conv2d(gray, Gx, mode='same')
    conv_y = conv2d(gray, Gy, mode='same')
    return np.sqrt(conv_x**2 + conv_y**2)


def apply_haar(image: np.ndarray, kernel_type: str = 'Haar12',
               scale: int = 1) -> np.ndarray:
    """
    对图像应用 Haar-like 特征滤波器。

    Parameters
    ----------
    image : np.ndarray
    kernel_type : str
    scale : int

    Returns
    -------
    result : np.ndarray
    """
    gray = _to_grayscale(image)
    kernel = haar_kernel(kernel_type, scale)
    return conv2d(gray, kernel, mode='same')
