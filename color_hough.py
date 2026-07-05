import cv2
import numpy as np
import time
from mss import mss

def get_lab_mask(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return cv2.inRange(lab, np.array([0, 0, 100]), np.array([255, 255, 255]))

def run_hough_node_detection():
    sct = mss()
    monitor = {"top": 450, "left": 250, "width": 600, "height": 600}
    
    prev_is_node = False
    print("=== Canny + 허프 변환 노드 인식 시작 ===")

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mask = get_lab_mask(frame)
        display = frame.copy()

        # [STEP 1] ROI 설정
        roi_size = 160
        h, w = mask.shape
        rx1, ry1 = (w//2 - roi_size//2), (h//2 - roi_size//2)
        rx2, ry2 = rx1 + roi_size, ry1 + roi_size
        roi_mask = mask[ry1:ry2, rx1:rx2]

        # [STEP 2] 전처리 (Blur -> Canny)
        # 가우시안 블러로 자잘한 노이즈 제거
        blurred = cv2.GaussianBlur(roi_mask, (5, 5), 0)
        # Canny로 외곽선만 추출 (이진화된 마스크이므로 threshold를 낮게 잡아도 됨)
        edges = cv2.Canny(blurred, 50, 150)

        # [STEP 3] 허프 변환 (edges 입력)
        # Canny를 거치면 선이 얇아지므로 threshold와 minLineLength를 조금 조정했습니다.
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=15)
        
        has_h = False
        has_v = False

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 가로/세로 분류 (기울기 기반)
                if abs(x2 - x1) > abs(y2 - y1): # 가로선
                    has_h = True
                    cv2.line(display, (x1+rx1, y1+ry1), (x2+rx1, y2+ry1), (255, 0, 0), 3)
                else: # 세로선
                    has_v = True
                    cv2.line(display, (x1+rx1, y1+ry1), (x2+rx1, y2+ry1), (0, 255, 0), 3)

        # [STEP 4] 교차점 판단
        is_node = has_h and has_v

        if is_node and not prev_is_node:
            print(f"[{time.strftime('%H:%M:%S')}] 교차점(NODE) 감지!")
        
        prev_is_node = is_node

        # [STEP 5] 시각화
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        status_text = "LOCKED" if is_node else "SEARCHING"
        cv2.putText(display, status_text, (rx1, ry1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if is_node else (0,255,0), 2)

        cv2.imshow('Hough Node Detector', display)
        cv2.imshow('Edges (Canny)', edges) # Canny 결과 확인용 창

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hough_node_detection()