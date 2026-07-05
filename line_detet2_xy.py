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
- 왼쪽(노란색), 오른쪽(보라색) 선을 구분하여 RANSAC 피팅
- 선을 새로 밟을 때마다 전역 좌표(pos_x, pos_y) 갱신
- 2초마다 터미널에 현재 상태 출력
=============================================================
"""

# 이진화 및 환경 설정
BIN_METHOD  = 2
THRESHOLD   = 127
BLOCK_SIZE  = 11
C           = 2
INVERT      = 0

# [추가] 전역 좌표 및 상태 변수
pos_x, pos_y = 0, 0      # 현재 전역 좌표 (격자 칸 단위)
is_on_left_line = False  # 왼쪽 선 감지 상태
is_on_right_line = False # 오른쪽 선 감지 상태

cap = cv2.VideoCapture('grid_test2.mp4')

\
# [수정] WINDOW_NORMAL을 사용하여 마우스로 창 크기 조절 가능하게 설정
cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 600, 400)

cv2.namedWindow('binary', cv2.WINDOW_NORMAL) # 이진화 창 크기 조절 가능
cv2.namedWindow('lines', cv2.WINDOW_NORMAL)  # 결과 창 크기 조절 가능

# 슬라이더 생성 (기존 유지)
cv2.createTrackbar('canny_low',    'controls', 50,  300, nothing)
cv2.createTrackbar('canny_high',   'controls', 150, 300, nothing)
cv2.createTrackbar('hough_thresh', 'controls', 50,  300, nothing)
cv2.createTrackbar('min_length',   'controls', 50,  300, nothing)
cv2.createTrackbar('max_gap',      'controls', 20,  100, nothing)
cv2.createTrackbar('ransac_thresh','controls', 15,  100, nothing)
cv2.createTrackbar('ransac_iter',   'controls', 100, 500, nothing)

last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    canny_low    = cv2.getTrackbarPos('canny_low',    'controls')
    canny_high   = cv2.getTrackbarPos('canny_high',   'controls')
    hough_thresh = cv2.getTrackbarPos('hough_thresh', 'controls')
    min_length   = cv2.getTrackbarPos('min_length',   'controls')
    max_gap      = cv2.getTrackbarPos('max_gap',      'controls')
    r_thresh     = cv2.getTrackbarPos('ransac_thresh','controls')
    r_iter       = cv2.getTrackbarPos('ransac_iter',  'controls')

    if r_iter < 2: r_iter = 2

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
    left_pts, right_pts = [], []

    if lines is not None:
        # 선분 끝점 수집 후 좌/우 분리 (기존 방식 유지)
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 1: continue
            
            mx = (x1 + x2) / 2
            if mx < w / 2: # 기울기를 통해 가로/세로 선 판단
                left_pts.extend([(x1, y1), (x2, y2)])
            else:
                right_pts.extend([(x1, y1), (x2, y2)])

        # 4단계: RANSAC fitting 및 좌표 갱신
        # [왼쪽 선 처리 - 노란색]
        if len(left_pts) >= 10: # 포인트가 10개 이상 모여야 진짜 선으로 인정
            if not is_on_left_line:
                pos_y += 1   # 새로운 선을 밟으면 좌표 증가 (예시로 Y축)
                is_on_left_line = True
            
            params_l = fit_ransac(np.array(left_pts), r_iter, r_thresh)
            if params_l:
                draw_ransac_line(result, *params_l, color=(0, 255, 255)) # 왼쪽: 노란색
        else:
            is_on_left_line = False

        # [오른쪽 선 처리 - 보라색]
        if len(right_pts) >= 10:
            if not is_on_right_line:
                pos_x += 1   # 새로운 선을 밟으면 좌표 증가 (예시로 X축)
                is_on_right_line = True
            
            params_r = fit_ransac(np.array(right_pts), r_iter, r_thresh)
            if params_r:
                draw_ransac_line(result, *params_r, color=(255, 0, 255)) # 오른쪽: 보라색
        else:
            is_on_right_line = False

    # 2초 주기 터미널 보고
    current_time = time.time()
    if current_time - last_print_time >= 1.0:
        print("-" * 50)
        print(f"[{time.strftime('%H:%M:%S')}] Global Position: X={pos_x}, Y={pos_y}")
        print(f"Detected - Left: {len(left_pts)//2} lines | Right: {len(right_pts)//2} lines")
        last_print_time = current_time

    # 정보 표시
    cv2.putText(edges,  bin_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(result, f"X:{pos_x} Y:{pos_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, f'RANSAC (t={r_thresh},i={r_iter})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    cv2.imshow('binary', edges)
    cv2.imshow('lines',  result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'):
        cv2.imwrite('edges_saved.png', edges)
        cv2.imwrite('ransac_saved.png', result)
        print('저장됨')

cap.release()
cv2.destroyAllWindows()