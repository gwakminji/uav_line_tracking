import cv2
import numpy as np
import time
from mss import mss

def get_lab_mask(frame):
    """
    LAB 색공간 기반 격자선 추출 함수
    조명 변화에 강인하게 L 채널 범위를 조정함
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # L(밝기): 0~150, A(초록-빨강): 0~255, B(파랑-노랑): 0~255
    return cv2.inRange(lab, np.array([0, 0, 0]), np.array([150, 255, 255]))

def run_square_cross_mission():
    # mss.MSS() 객체 생성 (Deprecation 경고 해결)
    with mss() as sct:
        # 시뮬레이터 창 위치 및 크기 설정
        monitor = {"top": 450, "left": 250, "width": 600, "height": 600}
        
        curr_x, curr_y = 0, 0
        prev_node = False
        origin_found = False
        
        print("=== Drone Grid Navigation System Started ===")

        while True:
            # 화면 캡처 및 전처리
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            mask = get_lab_mask(frame)
            display = frame.copy()

            # 1. ROI(관심 영역) 설정
            roi_size = 200 
            h, w = mask.shape
            rx1, ry1 = (w // 2 - roi_size // 2), (h // 2 - roi_size // 2)
            rx2, ry2 = rx1 + roi_size, ry1 + roi_size
            roi_mask = mask[ry1:ry2, rx1:rx2]
            rh, rw = roi_mask.shape

            # 2. 4방향 감지 센서 설정 (범위 및 민감도 최적화)
            pad = 40        # 감지 박스 두께 (선이 얇아도 잘 잡히도록 확장)
            threshold = 0.1 # 픽셀 점유율 기준 (10%만 차있어도 선으로 인지)
            
            top_box    = roi_mask[0:pad, :]
            bottom_box = roi_mask[rh-pad:rh, :]
            left_box   = roi_mask[:, 0:pad]
            right_box  = roi_mask[:, rw-pad:rw]

            has_top    = np.count_nonzero(top_box)    > (top_box.size * threshold)
            has_bottom = np.count_nonzero(bottom_box) > (bottom_box.size * threshold)
            has_left   = np.count_nonzero(left_box)   > (left_box.size * threshold)
            has_right  = np.count_nonzero(right_box)  > (right_box.size * threshold)

            # 연결된 방향 개수 계산
            line_count = sum([has_top, has_bottom, has_left, has_right])
            
            # 3. 원점(L-Corner) 인식 및 (0,0) 리셋 로직
            # 상/하 중 하나 AND 좌/우 중 하나만 연결된 경우 (ㄴ, ㄱ, ┌, ┘ 형태)
            is_l_shape = (line_count == 2) and ((has_top or has_bottom) and (has_left or has_right))
            
            if is_l_shape:
                origin_found = True
                curr_x, curr_y = 0, 0 # 원점 포착 시 좌표 초기화
                print("[RESET] Origin Point Detected. Coord set to (0,0)")

            # 4. 교차점 통과 시 좌표 업데이트 (Edge Trigger 방식)
            is_node = line_count >= 2
            if is_node and not prev_node:
                if origin_found and not is_l_shape: # 원점 이후 일반 교차점에서만 동작
                    # 축 판별을 위해 ROI 내부 가로/세로 픽셀량 비교
                    v_area = np.count_nonzero(roi_mask[:, rw//2-15:rw//2+15])
                    h_area = np.count_nonzero(roi_mask[rh//2-15:rh//2+15, :])

                    M = cv2.moments(roi_mask)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"]) # 무게중심 X
                        cy = int(M["m01"] / M["m00"]) # 무게중심 Y

                        # [X축 업데이트 - 전후 이동]
                        if v_area > h_area: 
                            # cy가 중앙보다 크면(아래쪽) 전진 중(+1), 작으면(위쪽) 하락 중(-1)
                            curr_x += 1 if cy > rh//2 else -1
                        # [Y축 업데이트 - 좌우 이동]
                        else: 
                            # cx가 중앙보다 작으면(왼쪽) 우측 이동 중(+1), 크면(오른쪽) 좌측 이동 중(-1)
                            curr_y += 1 if cx < rw//2 else -1
                        print(f"[NODE] Updated COORD: ({curr_x}, {curr_y})")

            prev_node = is_node

            # 5. 시각화 (UI 표시)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            
            # 상태 표시 문자열 설정
            state_txt = "ORIGIN (0,0)" if is_l_shape else ("FOUND" if origin_found else "SEARCHING")
            
            # COORD 글씨색 파란색으로 표시 (BGR: 255, 0, 0)
            cv2.putText(display, f"COORD: ({curr_x}, {curr_y})", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            
            cv2.putText(display, f"STATE: {state_txt}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 4방향 감지 센서 시각화 (초록: 인식, 빨강: 미인식)
            dots = [((w//2, ry1+15), has_top), ((w//2, ry2-15), has_bottom), 
                    ((rx1+15, h//2), has_left), ((rx2-15, h//2), has_right)]
            for pos, found in dots:
                cv2.circle(display, pos, 10, (0, 255, 0) if found else (0, 0, 255), -1)

            cv2.imshow('Grid Mapping System', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_square_cross_mission()