import cv2
import numpy as np
from t1 import binarize, nothing
from canny import canny_edge  # canny 단계 사용 시 주석 해제
from ransac import fit_ransac, draw_ransac_line
import time

"""
=============================================================
 파이프라인: 이진화 → Canny → HoughLinesP → RANSAC
=============================================================

허프만(HoughLinesP)이 검출한 선분 끝점들을 좌/우로 나눈 뒤,
각각 RANSAC으로 노이즈를 걸러내고 직선 하나씩 fitting.

=============================================================
"""

# ================================================================
# 이진화 설정 (코드에서 직접 수정)
# bin_method : 0=Global, 1=Adaptive, 2=Otsu
# threshold  : Global 전용 기준값 (0~255)
# block_size : Adaptive 전용 영역 크기 (홀수, 3 이상)
# C          : Adaptive 전용 보정값
# invert     : 0=그대로, 1=반전 (라인이 검정, 배경이 흰색인 경우)
BIN_METHOD  = 0
THRESHOLD   = 127
BLOCK_SIZE  = 11
C           = 2
INVERT      = 0
# ================================================================

cap = cv2.VideoCapture('grid_test.mp4')

cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 600, 400)
cv2.namedWindow('binary')
cv2.namedWindow('lines')

# Canny 슬라이더
cv2.createTrackbar('canny_low',    'controls', 50,  300, nothing)
cv2.createTrackbar('canny_high',   'controls', 150, 300, nothing)

# HoughLinesP 슬라이더
cv2.createTrackbar('hough_thresh', 'controls', 50,  300, nothing)
cv2.createTrackbar('min_length',   'controls', 50,  300, nothing)
cv2.createTrackbar('max_gap',      'controls', 20,  100, nothing)

# RANSAC 슬라이더
cv2.createTrackbar('ransac_thresh','controls', 15,  100, nothing)
cv2.createTrackbar('ransac_iter',  'controls', 100, 500, nothing)

# while 전 시간 초기화
last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # 슬라이더 파라미터 읽기
    canny_low    = cv2.getTrackbarPos('canny_low',    'controls')
    canny_high   = cv2.getTrackbarPos('canny_high',   'controls')
    hough_thresh = cv2.getTrackbarPos('hough_thresh', 'controls')
    min_length   = cv2.getTrackbarPos('min_length',   'controls')
    max_gap      = cv2.getTrackbarPos('max_gap',      'controls')
    r_thresh     = cv2.getTrackbarPos('ransac_thresh','controls')
    r_iter       = cv2.getTrackbarPos('ransac_iter',  'controls')

    if r_iter < 2:
        r_iter = 2

    # 1단계: 이진화
    binary, bin_name = binarize(gray, BIN_METHOD, THRESHOLD, BLOCK_SIZE, C)
    if INVERT == 1:
        binary = cv2.bitwise_not(binary)
        bin_name += ' [반전]'

    #2단계: Canny (주석 처리 시 이진화 결과를 바로 허프만에 사용)
    edges = canny_edge(binary, canny_low, canny_high)
    #edges = binary

    # 3단계: HoughLinesP — 선분 후보 수집
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh,
                            minLineLength=min_length, maxLineGap=max_gap)

    result = frame.copy()

    if lines is not None:
        # 선분 끝점 수집 후 좌/우 분리 (영상 중앙 기준)
        # 기울기 필터: 너무 수평인 선(|slope| < 0.3)은 노이즈로 제외
        left_pts, right_pts = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            #if abs(slope) < 0.3:
            #    continue
            mx = (x1 + x2) / 2 # 정확히 드론이 라인 가운데 있을 때만 동작 함 !!
            if mx < w / 2:
                left_pts.extend([(x1, y1), (x2, y2)])
            else:
                right_pts.extend([(x1, y1), (x2, y2)])

        # 4단계: RANSAC fitting 및 터미널 출력
        print("-" * 50)
        print(f"Frame Info - Left Pts: {len(left_pts)//2} lines | Right Pts: {len(right_pts)//2} lines")

        if len(left_pts) >= 2:
            params = fit_ransac(np.array(left_pts), r_iter, r_thresh)
            if params:
                # params는 보통 (slope, intercept) 또는 (vx, vy, x0, y0) 형태입니다.
                draw_ransac_line(result, *params, color=(0, 255, 255))  # 왼쪽: 노란색
                print(f"[LEFT] RANSAC Line Fitted: Params = {params}")
            else:
                print("[LEFT] RANSAC Failed: No Inliers found")

        if len(right_pts) >= 2:
            params = fit_ransac(np.array(right_pts), r_iter, r_thresh)
            if params:
                draw_ransac_line(result, *params, color=(255, 0, 255))  # 오른쪽: 보라색
                print(f"[RIGHT] RANSAC Line Fitted: Params = {params}")
            else:
                print("[RIGHT] RANSAC Failed: No Inliers found")


    cv2.putText(edges,  bin_name,                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(result, f'RANSAC (t={r_thresh},i={r_iter})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0),   2)

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
