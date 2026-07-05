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

허프만(HoughLinesP)이 검출한 선분 끝점들을 기울기에 따라 가로/세로로 나눈 뒤,
각각 RANSAC으로 노이즈를 걸러내고 직선 하나씩 fitting.
선을 통과할 때마다 전역 좌표(pos_x, pos_y)를 갱신함.

=============================================================
"""

# ================================================================
# 이진화 설정 (코드에서 직접 수정)
# bin_method : 0=Global, 1=Adaptive, 2=Otsu
# threshold  : Global 전용 기준값 (0~255)
# block_size : Adaptive 전용 영역 크기 (홀수, 3 이상)
# C          : Adaptive 전용 보정값
# invert     : 0=그대로, 1=반전 (라인이 검정, 배경이 흰색인 경우)
BIN_METHOD  = 2
THRESHOLD   = 127
BLOCK_SIZE  = 11
C           = 2
INVERT      = 0
# ================================================================

# [추가] 전역 좌표 및 상태 변수
pos_x, pos_y = 0, 0      # 현재 전역 좌표 (격자 칸 단위)
is_on_h_line = False     # 가로선 감지 상태 (중복 카운트 방지)
is_on_v_line = False     # 세로선 감지 상태 (중복 카운트 방지)

cap = cv2.VideoCapture('test.mp4')

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
cv2.createTrackbar('ransac_iter',   'controls', 100, 500, nothing)

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

    # [수정] 가로/세로 포인트 수집 리스트 생성
    h_pts, v_pts = [], []

    if lines is not None:
        # 선분 끝점 수집 후 기울기에 따라 가로/세로 분리
        for x1, y1, x2, y2 in lines[:, 0]:
            # 수직선 분모 0 에러 방지
            if abs(x2 - x1) < 1e-5:
                slope = 999
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # 기울기 필터: 가로/세로 그룹화
            if abs(slope) < 0.5:     # 가로선
                h_pts.extend([(x1, y1), (x2, y2)])
            elif abs(slope) > 1.5:   # 세로선
                v_pts.extend([(x1, y1), (x2, y2)])

        # 4단계: RANSAC fitting 및 좌표 갱신 로직
        # [가로선 처리]
        if len(h_pts) >= 10: # 포인트가 일정 이상일 때 선으로 간주
            if not is_on_h_line:
                pos_y += 1   # 선을 처음 밟을 때 Y 좌표 증가
                is_on_h_line = True
            
            params_h = fit_ransac(np.array(h_pts), r_iter, r_thresh)
            if params_h:
                draw_ransac_line(result, *params_h, color=(0, 255, 0)) # 가로: 초록색
        else:
            is_on_h_line = False # 선에서 완전히 벗어남

        # [세로선 처리]
        if len(v_pts) >= 10:
            if not is_on_v_line:
                pos_x += 1   # 선을 처음 밟을 때 X 좌표 증가
                is_on_v_line = True
            
            params_v = fit_ransac(np.array(v_pts), r_iter, r_thresh)
            if params_v:
                draw_ransac_line(result, *params_v, color=(0, 0, 255)) # 세로: 빨간색
        else:
            is_on_v_line = False

    # 2초 주기 터미널 출력 로직
    current_time = time.time()
    if current_time - last_print_time >= 2.0:
        print("-" * 50)
        print(f"[{time.strftime('%H:%M:%S')}] Global Position: X={pos_x}, Y={pos_y}")
        print(f"Frame Info - H Pts: {len(h_pts)//2} lines | V Pts: {len(v_pts)//2} lines")
        
        if 'params_h' in locals() and params_h:
            print(f"[HORIZ] RANSAC Params = {params_h}")
        if 'params_v' in locals() and params_v:
            print(f"[VERT] RANSAC Params = {params_v}")
        
        last_print_time = current_time

    # 화면에 정보 표시
    cv2.putText(edges,  bin_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    pos_text = f"Pos: X={pos_x}, Y={pos_y}"
    cv2.putText(result, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, f'RANSAC (t={r_thresh},i={r_iter})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

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