import cv2
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ──────────────────────────────────────────────
#  LAB 마스크 (구획선 추출)
# ──────────────────────────────────────────────
def get_lab_mask(frame,
                 l_low=0,   l_high=120,
                 rg_low=0,  rg_high=255,
                 by_low=0,  by_high=255):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lower = np.array([l_low,  rg_low,  by_low])
    upper = np.array([l_high, rg_high, by_high])
    return cv2.inRange(lab, lower, upper)


# ──────────────────────────────────────────────
#  중심선 추출
# ──────────────────────────────────────────────
def get_line_center(mask, frame_w):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=50,
                            minLineLength=60,
                            maxLineGap=20)
    if lines is None:
        return None, edges, lines

    cx_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cx_list.append((x1 + x2) // 2)

    center_x = int(np.mean(cx_list))
    return center_x, edges, lines


# ──────────────────────────────────────────────
#  ArUco 마커 인식 (원본 프레임)
# ──────────────────────────────────────────────
def detect_aruco(frame):
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(frame)

    results = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c  = corners[i][0]
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            results.append((int(marker_id), cx, cy))
    return results


# ──────────────────────────────────────────────
#  시각화 헬퍼
# ──────────────────────────────────────────────
def draw_lines(vis, lines, color=(0, 255, 0)):
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)

def draw_center(vis, center_x, color=(0, 0, 255)):
    if center_x is None:
        return
    h = vis.shape[0]
    cv2.line(vis, (center_x, 0), (center_x, h), color, 2)
    cv2.putText(vis, f'cx={center_x}', (center_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_aruco(vis, detections):
    for (marker_id, cx, cy) in detections:
        cv2.circle(vis, (cx, cy), 8, (255, 0, 255), -1)
        cv2.putText(vis,
                    f'ID:{marker_id} ({cx},{cy})',
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)


# ──────────────────────────────────────────────
#  마커 저장소
# ──────────────────────────────────────────────
detected_markers = {}   # { marker_id : (cx, cy) }

def update_markers(detections):
    for (marker_id, cx, cy) in detections:
        if marker_id not in detected_markers:
            print(f'[NEW] ArUco ID={marker_id}  pos=({cx}, {cy})')
        detected_markers[marker_id] = (cx, cy)

def print_all_markers():
    print('\n=== Saved ArUco Markers ===')
    for mid in sorted(detected_markers):
        x, y = detected_markers[mid]
        print(f'  ID={mid}  pos=({x}, {y})')
    print('===========================\n')


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────
if __name__ == '__main__':
    SOURCE = 'line_f.mp4'   # 카메라: SOURCE = 0

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open source: {SOURCE}')
        exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # color_seg.py 로 튜닝한 LAB 파라미터 입력
    LAB_L_LOW,  LAB_L_HIGH  = 0,   120
    LAB_RG_LOW, LAB_RG_HIGH = 0,   255
    LAB_BY_LOW, LAB_BY_HIGH = 0,   255

    cv2.namedWindow('original + ArUco', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask + Lines',     cv2.WINDOW_NORMAL)

    print('=== Line & ArUco Detection Start ===')
    print('q : quit  |  p : print saved markers\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # 1) 구획선 마스크
        mask = get_lab_mask(frame,
                            LAB_L_LOW,  LAB_L_HIGH,
                            LAB_RG_LOW, LAB_RG_HIGH,
                            LAB_BY_LOW, LAB_BY_HIGH)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        # 2) 중심선
        center_x, edges, lines = get_line_center(mask, frame_w)

        # 3) ArUco 인식 (원본)
        aruco_detections = detect_aruco(frame)
        update_markers(aruco_detections)

        # 4) 시각화
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        draw_lines(mask_vis, lines)
        draw_center(mask_vis, center_x)
        cv2.putText(mask_vis,
                    f'center_x = {center_x}' if center_x is not None else 'center_x = None',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        vis = frame.copy()
        draw_lines(vis, lines, color=(0, 255, 0))
        draw_center(vis, center_x, color=(0, 0, 255))
        draw_aruco(vis, aruco_detections)
        cv2.putText(vis,
                    f'Saved: {list(sorted(detected_markers.keys()))}',
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)

        cv2.imshow('original + ArUco', vis)
        cv2.imshow('mask + Lines',     mask_vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print_all_markers()

    # 루프 종료 후 1번만 출력
    print_all_markers()
    cap.release()
    cv2.destroyAllWindows()