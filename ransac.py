import cv2
import numpy as np

"""
RANSAC 라인 피팅 모듈

허프만이 검출한 선분들의 끝점을 모아서,
노이즈(outlier)를 무시하고 가장 잘 맞는 직선 하나를 fitting.

[ ransac_thresh ] : 직선으로부터 이 거리(픽셀) 이내면 inlier로 인정.
                    ↑ 높이면 → 더 많은 점을 inlier로 허용 (느슨하게 fitting)
                    ↓ 낮추면 → 딱 맞는 점만 inlier (엄격하게 fitting)
[ ransac_iter ]   : 반복 횟수. 높을수록 정확하지만 느려짐.
"""

def fit_ransac(points, n_iter, threshold):
    if len(points) < 2:
        return None

    best_inliers = []
    best_params  = None

    for _ in range(n_iter):
        idx = np.random.choice(len(points), 2, replace=False)
        x1, y1 = points[idx[0]]
        x2, y2 = points[idx[1]]

        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 1:
            continue

        # 각 점에서 이 직선까지의 수직 거리
        a, b, c = dy, -dx, dx * y1 - dy * x1
        dists    = np.abs(a * points[:, 0] + b * points[:, 1] + c) / length
        inliers  = np.where(dists < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params  = (x1, y1, dx, dy)

    if best_params is None or len(best_inliers) < 2:
        return None

    # inlier 점들로 최종 직선 재fitting
    inlier_pts = points[best_inliers].astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(vx), float(vy), float(x0), float(y0)


def draw_ransac_line(img, vx, vy, x0, y0, color=(255, 0, 255), thickness=3):
    h = img.shape[0]
    if abs(vy) < 1e-6:
        return
    t1 = (0 - y0) / vy
    t2 = (h - y0) / vy
    pt1 = (int(x0 + t1 * vx), 0)
    pt2 = (int(x0 + t2 * vx), h)
    cv2.line(img, pt1, pt2, color, thickness)
