import cv2
import numpy as np

"""
=============================================================
 색상 기반 전처리 (HSV / LAB)
=============================================================

[ color_method ] 슬라이더
    0 : HSV
    1 : LAB

[ HSV 파라미터 ]
    H_low / H_high  : 색상 범위 (0~180)
    S_low / S_high  : 채도 범위 (0~255)  낮으면 흰색/회색 포함
    V_low / V_high  : 명도 범위 (0~255)  낮으면 어두운 색 포함

    ※ 빨강은 H가 0~10 + 170~180 두 구간에 걸쳐있어서
       red_mode 슬라이더를 1로 올리면 자동으로 두 구간 합쳐줌

[ LAB 파라미터 ]
    L_low / L_high  : 밝기 범위       (0~255)
    RG_low / RG_high : 초록↔빨강 축   (0~255,  128이 중립)
    BY_low / BY_high : 파랑↔노랑 축   (0~255,  128이 중립)

    ※ 빨강 찾을 때 : RG_low 높이기 (128 이상)
       파랑 찾을 때 : BY_low 낮추기 (128 이하)

=============================================================
"""

def nothing(x):
    pass


def apply_hsv(frame, h_low, h_high, s_low, s_high, v_low, v_high, red_mode):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if red_mode == 1:
        # 빨강: 0~10 + 170~180 두 구간 합치기
        mask1 = cv2.inRange(hsv,
                            np.array([0,   s_low, v_low]),
                            np.array([10,  s_high, v_high]))
        mask2 = cv2.inRange(hsv,
                            np.array([170, s_low, v_low]),
                            np.array([180, s_high, v_high]))
        mask = cv2.bitwise_or(mask1, mask2)
        name = f'HSV RedMode S={s_low}~{s_high} V={v_low}~{v_high}'
    else:
        mask = cv2.inRange(hsv,
                           np.array([h_low,  s_low, v_low]),
                           np.array([h_high, s_high, v_high]))
        name = f'HSV H={h_low}~{h_high} S={s_low}~{s_high}'

    return mask, name


def apply_lab(frame, l_low, l_high, rg_low, rg_high, by_low, by_high):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mask = cv2.inRange(lab,
                       np.array([l_low, rg_low, by_low]),
                       np.array([l_high, rg_high, by_high]))
    name = f'LAB L={l_low}~{l_high} RG={rg_low}~{rg_high} BY={by_low}~{by_high}'

    return mask, name


if __name__ == '__main__':
    cap = cv2.VideoCapture('line_color.mp4')

    cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('controls', 600, 500)
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask',     cv2.WINDOW_NORMAL)
    cv2.namedWindow('result_color', cv2.WINDOW_NORMAL) 

    # 공통
    cv2.createTrackbar('color_method', 'controls', 0, 1, nothing)
    # 0 = HSV, 1 = LAB

    # HSV 슬라이더
    cv2.createTrackbar('H_low',  'controls', 0,   180, nothing)
    cv2.createTrackbar('H_high', 'controls', 180, 180, nothing)
    cv2.createTrackbar('S_low',  'controls', 50,  255, nothing)
    cv2.createTrackbar('S_high', 'controls', 255, 255, nothing)
    cv2.createTrackbar('V_low',  'controls', 50,  255, nothing)
    cv2.createTrackbar('V_high', 'controls', 255, 255, nothing)
    cv2.createTrackbar('red_mode','controls', 0,  1,   nothing)
    # 빨강 검출 시 1로

    # LAB 슬라이더
    cv2.createTrackbar('L_low',  'controls', 0,   255, nothing)
    cv2.createTrackbar('L_high', 'controls', 255, 255, nothing)
    cv2.createTrackbar('RG_low',  'controls', 0,   255, nothing)
    cv2.createTrackbar('RG_high', 'controls', 255, 255, nothing)
    cv2.createTrackbar('BY_low',  'controls', 0,   255, nothing)
    cv2.createTrackbar('BY_high', 'controls', 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        method   = cv2.getTrackbarPos('color_method', 'controls')

        h_low    = cv2.getTrackbarPos('H_low',   'controls')
        h_high   = cv2.getTrackbarPos('H_high',  'controls')
        s_low    = cv2.getTrackbarPos('S_low',   'controls')
        s_high   = cv2.getTrackbarPos('S_high',  'controls')
        v_low    = cv2.getTrackbarPos('V_low',   'controls')
        v_high   = cv2.getTrackbarPos('V_high',  'controls')
        red_mode = cv2.getTrackbarPos('red_mode','controls')

        l_low    = cv2.getTrackbarPos('L_low',   'controls')
        l_high   = cv2.getTrackbarPos('L_high',  'controls')
        rg_low   = cv2.getTrackbarPos('RG_low',  'controls')
        rg_high  = cv2.getTrackbarPos('RG_high', 'controls')
        by_low   = cv2.getTrackbarPos('BY_low',  'controls')
        by_high  = cv2.getTrackbarPos('BY_high', 'controls')

        if method == 0:
            mask, name = apply_hsv(frame,
                                   h_low, h_high,
                                   s_low, s_high,
                                   v_low, v_high,
                                   red_mode)
        else:
            mask, name = apply_lab(frame,
                                   l_low, l_high,
                                   rg_low, rg_high,
                                   by_low, by_high)

        # 마스크를 원본에 적용해서 해당 색상만 보이게
        result_color = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(mask,          name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        cv2.putText(result_color,  name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        method_label = "[ HSV ]" if method == 0 else "[ LAB ]"
        cv2.putText(mask, method_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)

        cv2.imshow('original', frame)          # 원본
        cv2.imshow('mask',     mask)           # 흑백 마스크 (흰색=검출된 색)
        cv2.imshow('result_color', result_color)   # 검출된 색상만 컬러로 표시

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('mask_saved.png',  mask)
            cv2.imwrite('color_saved.png', result_color)
            print(f'저장됨 - {name}')

    cap.release()
    cv2.destroyAllWindows()