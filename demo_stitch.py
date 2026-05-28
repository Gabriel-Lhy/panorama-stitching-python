"""
demo_stitch.py — Phase 2 演示：两图自动拼接
==============================================
完整 Pipeline: SIFT → FLANN+ratio test → RANSAC → Warp → Blend

用法:
    python demo_stitch.py                     # 用默认测试图
    python demo_stitch.py img1.jpg img2.jpg   # 用自己的图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.features import extract_sift, match_images, draw_keypoints, draw_matches
from src.stitcher import stitch_two_images


def main():
    # 参数
    if len(sys.argv) >= 3:
        img1_path, img2_path = sys.argv[1], sys.argv[2]
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "yard")
        img1_path = os.path.join(data_dir, "im01.jpg")
        img2_path = os.path.join(data_dir, "im02.jpg")

    print(f"Loading: {img1_path}")
    print(f"         {img2_path}")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("❌ Failed to load images. Run download_test_images.py first.")
        sys.exit(1)

    print(f"Image 1: {img1.shape[1]}×{img1.shape[0]}")
    print(f"Image 2: {img2.shape[1]}×{img2.shape[0]}")

    # ===================== 拼接 =====================
    print("\n--- Stitching (multi-band blend + exposure comp) ---")
    result, info = stitch_two_images(img1, img2, verbose=True)

    # ===================== 可视化 =====================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 0: 原始图 + 关键点
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Image 1 (reference)", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Image 2 (to warp)", fontsize=12)
    axes[0, 1].axis('off')

    # Inliers 匹配线
    inlier_indices = np.where(info['inliers'])[0]
    show_n = min(30, len(inlier_indices))
    vis_matches = draw_matches(
        img1, img2, info['sift1'], info['sift2'],
        info['match'], max_draw=show_n
    )
    axes[0, 2].imshow(cv2.cvtColor(vis_matches, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(
        f"Matches: {info['n_matches']} → {info['n_inliers']} inliers\n"
        f"Ratio: {info['inlier_ratio']:.1%}, Error: {info['mean_error']:.2f}px",
        fontsize=11
    )
    axes[0, 2].axis('off')

    # Row 1: 关键点 + 拼接结果
    kp1_vis = draw_keypoints(info['sift1'], max_points=300)
    axes[1, 0].imshow(cv2.cvtColor(kp1_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"SIFT Keypoints: {len(info['sift1'].keypoints)}", fontsize=12)
    axes[1, 0].axis('off')

    kp2_vis = draw_keypoints(info['sift2'], max_points=300)
    axes[1, 1].imshow(cv2.cvtColor(kp2_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"SIFT Keypoints: {len(info['sift2'].keypoints)}", fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(
        f"Stitched Result\n{result.shape[1]}×{result.shape[0]}",
        fontsize=14, fontweight='bold'
    )
    axes[1, 2].axis('off')

    plt.suptitle("Phase 2: Two-Image Auto Stitching Pipeline",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('stitch_demo.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: stitch_demo.png")
    plt.show()


if __name__ == '__main__':
    main()
