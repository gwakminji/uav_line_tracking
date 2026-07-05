import cv2
import numpy as np
import time

# --- [전처리 함수] color_seg.py의 LAB 방식 활용 ---
def get_lab_mask(frame, l_high=120):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # L_high 값을 조절하여 검정색 격자선만 추출
    mask = cv2.inRange(lab, np.array([0, 0, 0]), np.array([l_high, 255, 255]))
    return mask

# --- [시뮬레이션 환경 생성] 격자판 및 ArUco 배치 ---
def create_virtual_grid(offset_x, offset_y, w=640, h=480):
    # 배경 생성 (흰색 바닥)
    canvas = np.ones((h * 10, w * 10, 3), dtype=np.uint8) * 255
    grid_space = 200 # 격자 간격 (픽셀)
    
    # 격자선 그리기 (검정색)
    for i in range(0, canvas.shape[1], grid_space):
        cv2.line(canvas, (i, 0), (i, canvas.shape[0]), (0, 0, 0), 5)
    for j in range(0, canvas.shape[0], grid_space):
        cv2.line(canvas, (0, j), (canvas.shape[1], j), (0, 0, 0), 5)
        
    # 가상의 ArUco 마커 배치 (교차점 위치에 빨간 사각형으로 대체)
    # 실제 ArUco 검출은 원본에서 이루어지므로 여기서는 위치 확인용
    cv2.rectangle(canvas, (grid_space*2-20, grid_space*2-20), (grid_space*2+20, grid_space*2+20), (0, 0, 255), -1)

    # 드론 카메라 시야만큼 잘라내기 (이동 효과)
    view = canvas[offset_y:offset_y+h, offset_x:offset_x+w]
    return view

def run_simulation():
    # 설정 상수
    GRID_PIXEL_SIZE = 200 
    total_pixel_x = 0
    total_pixel_y = 0
    
    # 가상 드론 위치 (픽셀 단위)
    drone_pixel_x, drone_pixel_y = 500, 500 
    
    marker_log = []
    last_cy = None
    stop_until = 0

    while True:
        # 1. 가상 환경에서 프레임 가져오기 (드론이 조금씩 전진한다고 가정)
        if time.time() > stop_until:
            drone_pixel_y += 3 # 매 프레임 3픽셀씩 아래(전진)로 이동
            
        frame = create_virtual_grid(drone_pixel_x, drone_pixel_y)
        
        # 2. LAB 전처리 (color_seg 방식)
        mask = get_lab_mask(frame, l_high=120)
        
        # 3. 중심점 계산 및 논리적 좌표 카운팅
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cy = int(M["m01"] / M["m00"])
            
            if last_cy is not None and time.time() > stop_until:
                diff_y = cy - last_cy
                total_pixel_x += diff_y
            last_cy = cy

            # 반올림 좌표 계산
            logical_x = round(total_pixel_x / GRID_PIXEL_SIZE)
        else:
            logical_x = round(total_pixel_x / GRID_PIXEL_SIZE)

        # 4. 시각화 및 결과 확인
        display = frame.copy()
        h, w = frame.shape[:2]
        size = 200
        # 정사각형 ROI
        cv2.rectangle(display, (w//2-size//2, h//2-size//2), (w//2+size//2, h//2+size//2), (0, 255, 0), 2)
        
        cv2.putText(display, f"Logic X: {logical_x}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(display, f"Pixels: {total_pixel_x}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow('Virtual Simulation', display)
        cv2.imshow('LAB Mask', mask)

        # 키 입력 처리
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): # 수동 마커 인식 테스트 (S 누르면 2초 정지 및 저장)
            print(f"★ 마커 인식 이벤트 발생! 좌표 ({logical_x}, 0) 저장")
            stop_until = time.time() + 2.0
            marker_log.append([99, logical_x, 0])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_simulation()