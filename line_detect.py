import cv2
import numpy as np
from t1 import binarize, nothing
from canny import canny_edge  # canny 관련 함수. 필요 없을시, 여기만 지우면됨.

"""
=============================================================
 이진화 + 라인 감지 통합 파라미터 설명
=============================================================

[ 이진화 파라미터 ] ─ t1.py의 binarize() 함수 사용
    bin_method  : 0=Global, 1=Adaptive, 2=Otsu
    threshold   : Global 전용 기준값
    block_size  : Adaptive 전용 영역 크기
    C           : Adaptive 전용 보정값

[ line_method ] ─ 슬라이더 범위: 0 ~ 1
    라인 감지 방식 선택.
    0 : Hough             → 직선 전체를 무한히 연장해서 검출.
                            직선이 명확한 환경에 적합 (Global/Otsu 이진화와 잘 맞음).
    1 : Probabilistic     → 선분의 시작~끝점을 검출. Hough보다 빠르고 세밀함.
                            끊어진 선도 감지 가능 (Adaptive 이진화와 잘 맞음).

[ Hough 전용 파라미터 ] (line_method=0)
    hough_thresh : 직선으로 인정할 최소 투표 수.
                   ↑ 높이면 → 뚜렷한 긴 선만 검출 (노이즈 감소)
                   ↓ 낮추면 → 짧고 희미한 선도 검출 (노이즈 증가)

[ Probabilistic Hough 전용 파라미터 ] (line_method=1)
    min_length   : 선분으로 인정할 최소 길이 (픽셀).
                   ↑ 높이면 → 긴 선분만 검출
                   ↓ 낮추면 → 짧은 선분도 검출
    max_gap      : 선분 사이 허용 최대 빈 간격 (픽셀).
                   ↑ 높이면 → 끊어진 선을 하나로 이어서 검출
                   ↓ 낮추면 → 조금만 끊겨도 별개의 선으로 처리

=============================================================
"""

cap = cv2.VideoCapture('test.mp4')

cv2.namedWindow('controls')
cv2.namedWindow('binary')
cv2.namedWindow('lines')

# 이진화 슬라이더
cv2.createTrackbar('bin_method',  'controls', 0,   2,   nothing)
cv2.createTrackbar('threshold',   'controls', 127, 255, nothing)
cv2.createTrackbar('block_size',  'controls', 11,  51,  nothing)
cv2.createTrackbar('C',           'controls', 2,   20,  nothing)

# 라인 감지 슬라이더
cv2.createTrackbar('line_method', 'controls', 0,   1,   nothing)
cv2.createTrackbar('hough_thresh','controls', 100,  300, nothing)
cv2.createTrackbar('min_length',  'controls', 50,  300, nothing)
cv2.createTrackbar('max_gap',     'controls', 10,  100, nothing)
cv2.createTrackbar('invert',      'controls', 0,   1,   nothing)  # 1=반전
cv2.createTrackbar('canny_low',   'controls', 50,  300, nothing)  # canny 관련 함수. 필요 없을시, 여기만 지우면됨.
cv2.createTrackbar('canny_high',  'controls', 150, 300, nothing)  # canny 관련 함수. 필요 없을시, 여기만 지우면됨.

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이진화 파라미터 읽기
    bin_method  = cv2.getTrackbarPos('bin_method',  'controls')
    thresh_val  = cv2.getTrackbarPos('threshold',   'controls')
    block_size  = cv2.getTrackbarPos('block_size',  'controls')
    C           = cv2.getTrackbarPos('C',           'controls')

    # 라인 감지 파라미터 읽기
    line_method  = cv2.getTrackbarPos('line_method',  'controls')
    hough_thresh = cv2.getTrackbarPos('hough_thresh', 'controls')
    min_length   = cv2.getTrackbarPos('min_length',   'controls')
    max_gap      = cv2.getTrackbarPos('max_gap',      'controls')
    invert       = cv2.getTrackbarPos('invert',       'controls')

    # 이진화 (t1.py에서 import)
    binary, bin_name = binarize(gray, bin_method, thresh_val, block_size, C)

    # ─── 반전 처리 : 라인=검정, 배경=흰색인 경우 슬라이더 invert=1로 설정 ───
    if invert == 1:
        binary = cv2.bitwise_not(binary)
        bin_name += ' [반전]'

    canny_low  = cv2.getTrackbarPos('canny_low',  'controls')  # canny 관련 함수. 필요 없을시, 여기만 지우면됨.
    canny_high = cv2.getTrackbarPos('canny_high', 'controls')  # canny 관련 함수. 필요 없을시, 여기만 지우면됨.
    binary = canny_edge(binary, canny_low, canny_high)         # canny 관련 함수. 필요 없을시, 여기만 지우면됨.

    # 라인 감지 (원본 컬러 프레임 위에 선을 그림)
    result = frame.copy()

    if line_method == 0:
        lines = cv2.HoughLines(binary, 1, np.pi / 180, hough_thresh)
        line_name = f'Hough (thresh={hough_thresh})'
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(result, pt1, pt2, (0, 0, 255), 2)
    else:
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, hough_thresh,
                                minLineLength=min_length, maxLineGap=max_gap)
        line_name = f'Probabilistic (min={min_length}, gap={max_gap})'
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(binary, bin_name,  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    cv2.putText(result, line_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0),   2)

    cv2.imshow('binary',   binary)
    cv2.imshow('lines',    result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('binary_saved.png', binary)
        cv2.imwrite('lines_saved.png',  result)
        print(f'저장됨 - {bin_name} / {line_name}')

cap.release()
cv2.destroyAllWindows()
