"""
demo.py — 统一演示入口
=======================
用法:
    python demo.py --mode filters    滤波器可视化
    python demo.py --mode stitch     两图自动拼接
    python demo.py --mode compare    线性 vs 多频段对比
    python demo.py --mode all        按顺序运行全部
"""

import sys
import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Panoramic Image Stitching — Demo Runner"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["filters", "stitch", "compare", "all"],
        default="all",
        help="Which demo to run (default: all)"
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    demos = {
        "filters": ["demo_filters.py",  "Phase 1: Filter Kernels"],
        "stitch":  ["demo_stitch.py",   "Phase 2: Two-Image Stitch"],
        "compare": ["demo_blend_compare.py", "Phase 3: Blend Comparison"],
    }

    if args.mode == "all":
        modes = ["filters", "stitch", "compare"]
    else:
        modes = [args.mode]

    for i, mode in enumerate(modes):
        script, title = demos[mode]
        path = os.path.join(base, script)

        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, path],
            cwd=base,
            capture_output=False
        )
        if result.returncode != 0 and i < len(modes) - 1:
            print(f"\n⚠ {script} exited with code {result.returncode}")
            print("  Continue with next demo...\n")


if __name__ == "__main__":
    main()
