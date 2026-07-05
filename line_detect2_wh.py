import cv2
import numpy as np
from t1 import binarize, nothing
from canny import canny_edge
from ransac import fit_ransac, draw_ransac_line
import time

"""
=============================================================
 파이프라인: 이진화 → Canny → HoughLinesP → RANSAC
=============================================================
- 기울기(slope) 기준으로 가로선 / 세로선 분리
  가로선 → 보라색
  세로선 → 노란색
=============================================================
"""

BIN_METHOD  = 0
THRESHOLD   = 127
BLOCK_SIZE  = 10
C           = 2
INVERT      = 0

pos_x, pos_y = 0, 0
is_on_horizontal = False
is_on_vertical   = False

cap = cv2.VideoCapture('grid_test.mp4')

cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 600, 400)
cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
cv2.namedWindow('lines',  cv2.WINDOW_NORMAL)

cv2.createTrackbar('canny_low',     'controls', 50,  300, nothing)
cv2.createTrackbar('canny_high',    'controls', 150, 300, nothing)
cv2.createTrackbar('hough_thresh',  'controls', 50,  300, nothing)
cv2.createTrackbar('min_length',    'controls', 50,  300, nothing)
cv2.createTrackbar('max_gap',       'controls', 20,  100, nothing)
cv2.createTrackbar('ransac_thresh', 'controls', 15,  100, nothing)
cv2.createTrackbar('ransac_iter',   'controls', 100, 500, nothing)

# 가로/세로 구분 기울기 임계값 슬라이더 (0.1 단위, 실제값 = 슬라이더/10)
# 예: 슬라이더=5 → slope_thresh=0.5
cv2.createTrackbar('slope_thresh',  'controls', 5,   50,  nothing)

last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    canny_low    = cv2.getTrackbarPos('canny_low',     'controls')
    canny_high   = cv2.getTrackbarPos('canny_high',    'controls')
    hough_thresh = cv2.getTrackbarPos('hough_thresh',  'controls')
    min_length   = cv2.getTrackbarPos('min_length',    'controls')
    max_gap      = cv2.getTrackbarPos('max_gap',       'controls')
    r_thresh     = cv2.getTrackbarPos('ransac_thresh', 'controls')
    r_iter       = cv2.getTrackbarPos('ransac_iter',   'controls')
    slope_thresh = cv2.getTrackbarPos('slope_thresh',  'controls') / 10.0

    if r_iter < 2:
        r_iter = 2
    if slope_thresh <= 0:
        slope_thresh = 0.1

    # 1단계: 이진화
    binary, bin_name = binarize(gray, BIN_METHOD, THRESHOLD, BLOCK_SIZE, C)
    if INVERT == 1:
        binary = cv2.bitwise_not(binary)
        bin_name += ' [반전]'

    # 2단계: Canny
    edges = canny_edge(binary, canny_low, canny_high)

    # 3단계: HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh,
                            minLineLength=min_length, maxLineGap=max_gap)

    result = frame.copy()
    horizontal_pts = []   # 가로선 점들 → 보라색
    vertical_pts   = []   # 세로선 점들 → 노란색

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1

            # 기울기 계산
            if abs(dx) < 1:
                slope = float('inf')   # 수직선
            else:
                slope = abs(dy / dx)

            # ── 기울기로 가로/세로 분류 ──────────────────────────
            # slope < thresh  → 수평에 가까움 → 가로선
            # slope >= thresh → 수직에 가까움 → 세로선
            if slope < slope_thresh:
                horizontal_pts.extend([(x1, y1), (x2, y2)])
            else:
                vertical_pts.extend([(x1, y1), (x2, y2)])

    # 4단계: RANSAC fitting
    # 가로선 → 보라색
    if len(horizontal_pts) >= 4:
        if not is_on_horizontal:
            pos_y += 1
            is_on_horizontal = True
        params_h = fit_ransac(np.array(horizontal_pts), r_iter, r_thresh)
        if params_h:
            draw_ransac_line(result, *params_h, color=(255, 0, 255))   # 보라색
    else:
        is_on_horizontal = False

    # 세로선 → 노란색
    if len(vertical_pts) >= 4:
        if not is_on_vertical:
            pos_x += 1
            is_on_vertical = True
        params_v = fit_ransac(np.array(vertical_pts), r_iter, r_thresh)
        if params_v:
            draw_ransac_line(result, *params_v, color=(0, 255, 255))   # 노란색
    else:
        is_on_vertical = False

    # 터미널 출력 (1초 주기)
    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        print("-" * 50)
        print(f"[{time.strftime('%H:%M:%S')}] Global Position: X={pos_x}, Y={pos_y}")
        print(f"가로선 점 수: {len(horizontal_pts)//2}  |  세로선 점 수: {len(vertical_pts)//2}")
        print(f"slope_thresh: {slope_thresh:.1f}")
        last_print_time = current_time

    # 화면 정보 표시
    cv2.putText(edges,  bin_name,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(result, f"X:{pos_x}  Y:{pos_y}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, f"slope_thresh={slope_thresh:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0),   2)
    cv2.putText(result, "가로=보라  세로=노랑",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow('binary', edges)
    cv2.imshow('lines',  result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('edges_saved.png',  edges)
        cv2.imwrite('ransac_saved.png', result)
        print('저장됨')

cap.release()
cv2.destroyAllWindows()