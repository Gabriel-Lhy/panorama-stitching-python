"""
demo_filters.py — Phase 1 演示脚本
====================================
运行所有滤波器（Gaussian / Sobel / Haar-like）并保存对比图。
不需要额外下载图片，使用 scipy 内置的测试图。

用法:
    python demo_filters.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from src.filters import (
    apply_gaussian, apply_sobel, apply_haar,
    gaussian_kernel, sobel_kernels, haar_kernel, set_scale
)


def main():
    # 使用 scipy 内置测试图
    image = datasets.ascent()

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Value range: [{image.min()}, {image.max()}]")

    # ---------------------------------------------------------------
    # 1. Gaussian 平滑（不同 kernel_size）
    # ---------------------------------------------------------------
    gauss_3 = apply_gaussian(image, kernel_size=3)
    gauss_7 = apply_gaussian(image, kernel_size=7)

    # ---------------------------------------------------------------
    # 2. Sobel 边缘检测（不同 scale）
    # ---------------------------------------------------------------
    sobel_1 = apply_sobel(image, scale=1)
    sobel_4 = apply_sobel(image, scale=4)

    # ---------------------------------------------------------------
    # 3. Haar-like 特征（全部 5 种核）
    # ---------------------------------------------------------------
    haar_types = ['Haar12', 'Haar21', 'Haar13', 'Haar31', 'Haar22']
    haar_results = {}
    for ht in haar_types:
        haar_results[ht] = apply_haar(image, kernel_type=ht, scale=4)

    # ---------------------------------------------------------------
    # 可视化
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 0: Gaussian
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gauss_3, cmap='gray')
    axes[0, 1].set_title('Gaussian (k=3)', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gauss_7, cmap='gray')
    axes[0, 2].set_title('Gaussian (k=7)', fontsize=12)
    axes[0, 2].axis('off')

    # 显示 3×3 高斯核
    gk = gaussian_kernel(3)
    axes[0, 3].imshow(gk, cmap='Blues')
    for i in range(3):
        for j in range(3):
            axes[0, 3].text(j, i, f'{gk[i,j]:.2f}', ha='center', va='center', fontsize=8)
    axes[0, 3].set_title('Gaussian Kernel 3×3', fontsize=12)
    axes[0, 3].axis('off')

    # Row 1: Sobel
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].set_title('Original', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sobel_1, cmap='gray')
    axes[1, 1].set_title('Sobel (scale=1)', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(sobel_4, cmap='gray')
    axes[1, 2].set_title('Sobel (scale=4)', fontsize=12)
    axes[1, 2].axis('off')

    # 显示 Sobel X 核
    Gx, Gy = sobel_kernels()
    axes[1, 3].imshow(Gx, cmap='RdBu', vmin=-2, vmax=2)
    for i in range(3):
        for j in range(3):
            axes[1, 3].text(j, i, f'{Gx[i,j]:.0f}', ha='center', va='center', fontsize=9)
    axes[1, 3].set_title('Sobel Gx', fontsize=12)
    axes[1, 3].axis('off')

    # Row 2: Haar-like（选 4 个效果最好的）
    haar_show = ['Haar12', 'Haar21', 'Haar22', 'Haar13']
    for idx, ht in enumerate(haar_show):
        ax = axes[2, idx]
        ax.imshow(haar_results[ht], cmap='RdBu', vmin=-np.abs(haar_results[ht]).max(),
                  vmax=np.abs(haar_results[ht]).max())
        ax.set_title(f'{ht} (scale=4)', fontsize=12)
        ax.axis('off')

    plt.suptitle('Phase 1: Filter Demo — Gaussian · Sobel · Haar-like',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('filter_demo.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: filter_demo.png")
    plt.show()


if __name__ == '__main__':
    main()
