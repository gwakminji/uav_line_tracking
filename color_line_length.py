import cv2
import numpy as np
import time
from mss import mss

def get_lab_mask(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return cv2.inRange(lab, np.array([51, 0, 0]), np.array([255, 255, 255]))

def run_junction_mission():
    sct = mss()
    # [설정] 시뮬레이터 창 위치 (본인 환경에 맞게 조정 필수)
    monitor = {"top": 450, "left": 250, "width": 600, "height": 600}
    
    curr_x, curr_y = 0, 0
    prev_is_node = False
    stop_until = 0

    print("=== 분기점(Node) 인식 강화 시스템 가동 ===")

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mask = get_lab_mask(frame)
        display = frame.copy()

        # [STEP 1] ROI 설정 (160x160)
        roi_size = 160
        h, w = mask.shape
        rx1, ry1 = (w//2 - roi_size//2), (h//2 - roi_size//2)
        rx2, ry2 = rx1 + roi_size, ry1 + roi_size
        roi_mask = mask[ry1:ry2, rx1:rx2]

        # [STEP 2] 허프 변환 파라미터 완화 (Y축 인식 살리기)
        # minLineLength를 20으로 낮추고 maxLineGap을 20으로 늘려서 짧거나 끊긴 선도 잡음
        lines = cv2.HoughLinesP(roi_mask, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=20)
        
        h_lines, v_lines = [], []
        h_len_total, v_len_total = 0, 0
        h_y_list, v_x_list = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if abs(x2 - x1) > abs(y2 - y1): # 가로선 (Blue)
                    h_lines.append(line)
                    h_len_total += length
                    h_y_list.append((y1 + y2) / 2)
                    cv2.line(display, (x1+rx1, y1+ry1), (x2+rx1, y2+ry1), (255, 0, 0), 3)
                else: # 세로선 (Green)
                    v_lines.append(line)
                    v_len_total += length
                    v_x_list.append((x1 + x2) / 2)
                    cv2.line(display, (x1+rx1, y1+ry1), (x2+rx1, y2+ry1), (0, 255, 0), 3)

        # [STEP 3] 노드 감지 로직 완화 (Pure Junction 감지)
        # 스크린샷처럼 한쪽 선이 짧더라도 둘 다 존재하면 노드로 인정
        has_h = len(h_lines) >= 1
        has_v = len(v_lines) >= 1
        is_node = has_h and has_v

        # [STEP 4] 좌표 업데이트 (Edge Trigger 및 방향성 부여)
        if is_node and not prev_is_node:
            if time.time() > stop_until:
                # 1. 이동 축 판별 (지배적 길이 + 순간 변화량 조합)
                
                # 스크린샷 상황: v_len_total이 더 크더라도 Y가 바뀌어야 함. 
                # 이를 위해 Y축 판단 임계값을 낮추거나 순간적인 h_len_total 변화를 감지해야 함.
                # 임시로 Y축 인식을 위해 h_len_total에 가중치를 둠
                
                if v_len_total > (h_len_total * 1.5): # 세로선이 압도적일 때만 X축 이동으로 간주
                    # X축 부호 판별: 선이 아래쪽(avg_h_y > roi_size/2)에서 나타나면 증가
                    avg_h_y = np.mean(h_y_list)
                    curr_x += 1 if avg_h_y > (roi_size / 2) else -1
                    print(f"X Updated: {curr_x}")
                
                elif h_len_total >= 20: # 가로선이 조금이라도 보이면 Y축 이동으로 우선 판정
                    avg_v_x = np.mean(v_x_list)
                    # Y축 부호 판별: 선이 오른쪽에서 나타나 왼쪽으로 흐르면 Y 증가
                    # 시뮬레이터 방향에 맞게 < 를 > 로 바꿀 수 있음
                    curr_y += 1 if avg_v_x < (roi_size / 2) else -1
                    print(f"Y Updated: {curr_y}")

        prev_is_node = is_node

        # [STEP 5] 시각화 및 정보 출력
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(display, f"POS: ({curr_x}, {curr_y})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        status_text = "NODE_LOCKED" if is_node else "MOVING"
        status_color = (0, 0, 255) if is_node else (0, 255, 0)
        cv2.putText(display, status_text, (rx1, ry1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow('Pure Vision Grid Tracker (Enhanced Junction)', display)
        cv2.imshow('Full LAB Mask', mask) # 전체 마스크 모니터링용

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_junction_mission()