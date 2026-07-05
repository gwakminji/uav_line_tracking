import cv2
import numpy as np
from color_seg import apply_hsv # 기존 color_seg.py에서 함수 로드

class GatingNavigation:
    def __init__(self):
        self.grid_length = 0.0      # 학습된 격자 간격
        self.is_calibrated = False  # 학습 완료 여부
        self.last_node_x = 0.0      # 마지막 리셋 좌표
        self.gate_range = 0.15      # 게이트 범위 (15%)
        self.node_count = 0

    def check_intersection(self, mask):
        """가로선 밀집도 기반 꼭짓점 판단"""
        h, w = mask.shape
        # 중앙 가로 한 줄의 흰색 픽셀 수 계산
        center_line = mask[h//2, :]
        white_pixels = np.sum(center_line == 255)
        return white_pixels > (w * 0.45) # 화면 가로의 45% 이상이 선이면 교차점

    def run(self, frame, current_dist):
        # 1. 전처리 (color_seg.py 함수 사용)
        mask, _ = apply_hsv(frame, 0, 180, 50, 255, 50, 255, red_mode=1)
        
        dist_from_last = abs(current_dist - self.last_node_x)
        status = "CRUISING"

        # 2. 학습 모드
        if not self.is_calibrated:
            if self.check_intersection(mask):
                if self.last_node_x != 0:
                    self.grid_length = dist_from_last
                    self.is_calibrated = True
                self.last_node_x = current_dist
            return "CALIBRATING", mask

        # 3. 게이팅 및 리셋 모드
        lower = self.grid_length * (1 - self.gate_range)
        upper = self.grid_length * (1 + self.gate_range)

        if lower <= dist_from_last <= upper:
            status = "GATE OPEN"
            if self.check_intersection(mask):
                self.last_node_x = current_dist # 좌표 강제 업데이트(리셋)
                self.node_count += 1
                status = "NODE RESET!"
        
        return status, mask

# 실행부
cap = cv2.VideoCapture('line_color.mp4')
nav = GatingNavigation()
fake_t265_dist = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    fake_t265_dist += 0.02 # 프레임당 2cm 이동한다고 가정
    status, mask = nav.run(frame, fake_t265_dist)

    # 결과 표시
    cv2.putText(frame, f"Dist: {fake_t265_dist:.2f}m", (10, 30), 1, 1.5, (255,255,0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 70), 1, 1.5, (0,255,0), 2)
    if nav.is_calibrated:
        cv2.putText(frame, f"Grid: {nav.grid_length:.2f}m", (10, 110), 1, 1.5, (0,0,255), 2)

    cv2.imshow('Main', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()