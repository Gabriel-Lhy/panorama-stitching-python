"""
demo_blend_compare.py — 线性混合 vs 多频段融合 对比演示
==========================================================
直观对比两种融合方式的差异，是简历上最容易出"哇塞"效果的对比图。

用法:
    python demo_blend_compare.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.features import extract_sift, draw_keypoints, draw_matches
from src.stitcher import stitch_two_images


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "yard")
    img1_path = os.path.join(data_dir, "im01.jpg")
    img2_path = os.path.join(data_dir, "im02.jpg")

    print(f"Loading: {img1_path}\n         {img2_path}")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("❌ Run download_test_images.py first")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  Linear Blend (原 MATLAB 50/50 平均)")
    print(f"{'='*50}")
    result_linear, info_l = stitch_two_images(
        img1, img2, blend_mode='linear',
        exposure_compensation=False, verbose=True
    )

    print(f"\n{'='*50}")
    print("  Multi-band Blend + Exposure Comp")
    print(f"{'='*50}")
    result_mb, info_mb = stitch_two_images(
        img1, img2, blend_mode='multiband',
        exposure_compensation=True, verbose=True
    )

    # ===================== 可视化对比 =====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # 线性混合
    axes[0, 0].imshow(cv2.cvtColor(result_linear, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(
        "Linear Blend (50/50)\n"
        f"inliers: {info_l['n_inliers']}/{info_l['n_matches']} "
        f"({info_l['inlier_ratio']:.1%})",
        fontsize=13, color='#c0392b'
    )
    axes[0, 0].axis('off')

    # 多频段融合
    axes[0, 1].imshow(cv2.cvtColor(result_mb, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(
        "Multi-band Blend\n"
        f"+ Exposure Compensation",
        fontsize=13, color='#27ae60'
    )
    axes[0, 1].axis('off')

    # 差异放大图
    # 找到两张结果图的共同区域
    h = min(result_linear.shape[0], result_mb.shape[0])
    w = min(result_linear.shape[1], result_mb.shape[1])
    diff = np.abs(
        result_linear[:h, :w].astype(np.float32) -
        result_mb[:h, :w].astype(np.float32)
    )
    # 放大 5 倍差异以便观察
    diff_enhanced = np.clip(diff * 5, 0, 255).astype(np.uint8)

    axes[1, 0].imshow(diff_enhanced)
    axes[1, 0].set_title(
        "Difference Map (×5 enhanced)\n"
        f"mean diff = {diff.mean():.1f} px\n"
        "bright regions = where blending differs most",
        fontsize=12
    )
    axes[1, 0].axis('off')

    # 匹配线可视化
    show_n = min(40, info_mb['n_matches'])
    vis_matches = draw_matches(
        img1, img2, info_mb['sift1'], info_mb['sift2'],
        info_mb['match'], max_draw=show_n
    )
    axes[1, 1].imshow(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"Feature Matches\n"
        f"{info_mb['n_matches']} total → {info_mb['n_inliers']} RANSAC inliers\n"
        f"mean error = {info_mb['mean_error']:.2f}px",
        fontsize=12
    )
    axes[1, 1].axis('off')

    plt.suptitle(
        "Phase 3: Linear Blend vs Multi-band Blend Comparison",
        fontsize=16, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('blend_compare.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: blend_compare.png")
    plt.show()


if __name__ == '__main__':
    main()
