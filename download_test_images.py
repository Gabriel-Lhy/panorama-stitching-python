"""
download_test_images.py — 从原项目下载测试图片
================================================
用法:
    python download_test_images.py
"""

import os
import urllib.request

BASE = "https://raw.githubusercontent.com/YICHENG-LAI/Panoramic-Image-Stitching/master"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# (源文件夹, 源文件名, 本地子目录)
FILES = [
    ("assg1.1%20Kernel", "im.jpg", ""),
    ("assg1.5%20Stitch", "im01.jpg", "yard"),
    ("assg1.5%20Stitch", "im02.jpg", "yard"),
    ("assg1.6%20Stitch%20multi-images", "im01.jpg", "yard5"),
    ("assg1.6%20Stitch%20multi-images", "im02.jpg", "yard5"),
    ("assg1.6%20Stitch%20multi-images", "im03.jpg", "yard5"),
    ("assg1.6%20Stitch%20multi-images", "im04.jpg", "yard5"),
    ("assg1.6%20Stitch%20multi-images", "im05.jpg", "yard5"),
    ("assg1.7%20unordered%20images", "M1.jpg", "campus"),
    ("assg1.7%20unordered%20images", "M2.jpg", "campus"),
    ("assg1.7%20unordered%20images", "M3.jpg", "campus"),
    ("assg1.7%20unordered%20images", "M4.jpg", "campus"),
    ("assg1.7%20unordered%20images", "M5.jpg", "campus"),
]

def main():
    for folder, fname, subdir in FILES:
        target_dir = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
        os.makedirs(target_dir, exist_ok=True)

        url = f"{BASE}/{folder}/{fname}"
        outpath = os.path.join(target_dir, fname)

        if os.path.exists(outpath):
            print(f"⏭  Skipped: {subdir}/{fname}" if subdir else f"⏭  Skipped: {fname}")
            continue

        try:
            urllib.request.urlretrieve(url, outpath)
            size_kb = os.path.getsize(outpath) / 1024
            label = f"{subdir}/{fname}" if subdir else fname
            print(f"✅ {label} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"❌ Failed: {fname} — {e}")

    print(f"\nDone. Images in: {DATA_DIR}")


if __name__ == "__main__":
    main()
