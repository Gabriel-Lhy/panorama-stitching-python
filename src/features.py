"""
features.py — SIFT 特征提取与匹配
===================================
用 OpenCV 的 SIFT 替代原项目的外部 C 可执行文件 (siftWin32)。
匹配策略升级：FLANN + Lowe's ratio test，比原项目的简单最近邻更稳健。

原 MATLAB 对应：
  sift.m, find_all_matches.m, showkeys.m, showdescriptor.m

改进：
  - cv2.SIFT_create() 替代 tmp.pgm → siftWin32 → tmp.key 的管道
  - FLANN (Fast Library for Approximate Nearest Neighbors) 替代暴力匹配
  - Lowe's ratio test 过滤错误匹配（原项目没做）
  - 返回命名元组，而非松散数组
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, NamedTuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class SIFTResult(NamedTuple):
    """SIFT 提取结果"""
    keypoints: List[cv2.KeyPoint]   # 关键点列表（OpenCV KeyPoint 对象）
    descriptors: np.ndarray          # 描述子矩阵，shape (N, 128)，dtype=float32
    image: np.ndarray                # 灰度图像


class MatchResult(NamedTuple):
    """特征匹配结果"""
    src_pts: np.ndarray    # 源图像匹配点坐标，shape (M, 2)
    dst_pts: np.ndarray    # 目标图像匹配点坐标，shape (M, 2)
    good_matches: list     # OpenCV DMatch 列表（ratio test 通过的）
    all_matches: list      # 全部匹配（未过滤）


# ---------------------------------------------------------------------------
# SIFT 特征提取
# ---------------------------------------------------------------------------

def extract_sift(image: np.ndarray,
                 nfeatures: int = 0,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0) -> SIFTResult:
    """
    从图像中提取 SIFT 关键点和描述子。

    原 MATLAB sift.m 将图像写入 PGM 临时文件，调用外部 C 可执行文件，
    再解析输出文件。这里直接用 OpenCV 内置实现，零 I/O 开销。

    Parameters
    ----------
    image : np.ndarray
        BGR 或灰度图像
    nfeatures : int
        保留的最佳特征数量，0 表示不限
    contrast_threshold : float
        对比度阈值，越小则保留更多弱特征（默认 0.04）
    edge_threshold : float
        边缘阈值，越大则保留更多边缘特征（默认 10）

    Returns
    -------
    SIFTResult
        .keypoints  : OpenCV KeyPoint 对象列表
        .descriptors: (N, 128) float32 描述子
        .image      : 灰度图像
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return SIFTResult(
        keypoints=keypoints,
        descriptors=descriptors,
        image=gray
    )


# ---------------------------------------------------------------------------
# 特征匹配
# ---------------------------------------------------------------------------

def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   ratio_thresh: float = 0.75,
                   cross_check: bool = True) -> MatchResult:
    """
    FLANN 特征匹配 + Lowe's ratio test。

    原 MATLAB find_all_matches.m 的做法：
      1. 对 im2 的每个描述子，在 im1 中找最近邻
      2. 保证一对一匹配（冲突时保留距离更近的）
    没有 ratio test，这意味着大量错误匹配会混进来。

    这里的改进：
      1. FLANN 索引 → 对每个描述子找 2 个最近邻
      2. Lowe's ratio test: d1/d2 < ratio_thresh 才保留
      3. 可选双向匹配（cross-check）

    Parameters
    ----------
    desc1, desc2 : np.ndarray
        描述子矩阵，每行一个 128 维向量
    ratio_thresh : float
        Lowe's ratio: 第一近邻距离 / 第二近邻距离 的阈值
        0.75 是 SIFT 原论文推荐值
    cross_check : bool
        是否做双向匹配验证

    Returns
    -------
    MatchResult
        .src_pts, .dst_pts : 匹配点坐标（需要调用者填入实际坐标）
        .good_matches       : FLANN 匹配结果
        .all_matches        : ratio test 前的全部匹配
    """
    # 描述子需要是 float32
    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)

    # FLANN 参数（针对 SIFT 描述子优化）
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 对每个描述子找 2 个最近邻
    matches_raw = flann.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches_raw:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 双向匹配验证
    if cross_check:
        matches_rev = flann.knnMatch(desc2, desc1, k=2)
        good_rev = []
        for m, n in matches_rev:
            if m.distance < ratio_thresh * n.distance:
                good_rev.append(m)

        # 只保留双向都匹配的点对
        rev_map = {m.trainIdx: m.queryIdx for m in good_rev}
        good_matches = [m for m in good_matches
                        if m.queryIdx in rev_map
                        and rev_map[m.queryIdx] == m.trainIdx]

    # 展平为一维列表
    all_raw = [m for m, n in matches_raw]

    return MatchResult(
        src_pts=np.array([]),
        dst_pts=np.array([]),
        good_matches=good_matches,
        all_matches=all_raw
    )


def match_images(img1: np.ndarray, img2: np.ndarray,
                 ratio_thresh: float = 0.75,
                 nfeatures: int = 0) -> Tuple[SIFTResult, SIFTResult, MatchResult]:
    """
    端到端：两图 SIFT 提取 + 匹配。

    这是最常用的入口，一次调用搞定所有。

    Parameters
    ----------
    img1, img2 : np.ndarray
        输入图像（BGR 或灰度）
    ratio_thresh : float
        Lowe's ratio 阈值
    nfeatures : int
        每张图保留的最大特征数

    Returns
    -------
    (sift1, sift2, match) : (SIFTResult, SIFTResult, MatchResult)
    """
    sift1 = extract_sift(img1, nfeatures=nfeatures)
    sift2 = extract_sift(img2, nfeatures=nfeatures)

    match = match_features(sift1.descriptors, sift2.descriptors,
                           ratio_thresh=ratio_thresh)

    # 将 FLANN 的索引映射回实际像素坐标
    kp1 = sift1.keypoints
    kp2 = sift2.keypoints

    src_pts = np.float32([kp2[m.trainIdx].pt for m in match.good_matches])
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in match.good_matches])

    match = MatchResult(
        src_pts=src_pts,
        dst_pts=dst_pts,
        good_matches=match.good_matches,
        all_matches=match.all_matches
    )

    return sift1, sift2, match


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def draw_keypoints(sift_result: SIFTResult, max_points: int = 500,
                   color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    在图像上绘制 SIFT 关键点（含方向和尺度）。

    Parameters
    ----------
    sift_result : SIFTResult
    max_points : int
        最多绘制多少关键点（避免图太密）
    color : tuple
        BGR 颜色

    Returns
    -------
    vis : np.ndarray (BGR)
    """
    kp = sift_result.keypoints[:max_points]
    vis = cv2.drawKeypoints(
        sift_result.image, kp, None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return vis


def draw_matches(img1: np.ndarray, img2: np.ndarray,
                 sift1: SIFTResult, sift2: SIFTResult,
                 match: MatchResult, max_draw: int = 50) -> np.ndarray:
    """
    并排绘制两图及匹配线。

    Parameters
    ----------
    img1, img2 : np.ndarray
    sift1, sift2 : SIFTResult
    match : MatchResult
    max_draw : int
        最多绘制多少条匹配线

    Returns
    -------
    vis : np.ndarray (BGR)
    """
    h1, w1 = sift1.image.shape[:2]
    h2, w2 = sift2.image.shape[:2]

    # 创建并排画布
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = cv2.cvtColor(sift1.image, cv2.COLOR_GRAY2BGR) if sift1.image.ndim == 2 else img1
    vis[:h2, w1:w1+w2] = cv2.cvtColor(sift2.image, cv2.COLOR_GRAY2BGR) if sift2.image.ndim == 2 else img2

    # 随机选 max_draw 条匹配线
    indices = list(range(len(match.good_matches)))
    if len(indices) > max_draw:
        import random
        indices = random.sample(indices, max_draw)

    for i in indices:
        m = match.good_matches[i]
        pt1 = tuple(map(int, sift1.keypoints[m.queryIdx].pt))
        pt2 = tuple(map(int, sift2.keypoints[m.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])

        # 随机颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)

    return vis
