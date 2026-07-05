import cv2
import numpy as np

"""
현 코드는 흑백을 기준으로 되어있는 코드 
    즉, 밝기를 기준으로 나뉘기 때문에 흰/검 이런식은 괜찮지만,
    '밝기가 비슷한데 색상만 다른 경우'엔 사용 불가.

라인이 검정, 배경이 흰색이라면 '반전' 시키는 코드 추가 필요.

"""

def nothing(x):
    pass


def binarize(gray, method, thresh_val, block_size, C):
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    if method == 0:
        _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        name = f'Global (thresh={thresh_val})'
    elif method == 1:
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C)
        name = f'Adaptive (block={block_size}, C={C})'
    else:
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        name = 'Otsu'

    return result, name


if __name__ == '__main__':
    cap = cv2.VideoCapture('test.mp4')
    cv2.namedWindow('result')

    cv2.createTrackbar('method',     'result', 0,   2,  nothing)
    cv2.createTrackbar('threshold',  'result', 127, 255, nothing)
    cv2.createTrackbar('block_size', 'result', 11,  51,  nothing)
    cv2.createTrackbar('C',          'result', 2,   20,  nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        method     = cv2.getTrackbarPos('method',     'result')
        thresh_val = cv2.getTrackbarPos('threshold',  'result')
        block_size = cv2.getTrackbarPos('block_size', 'result')
        C          = cv2.getTrackbarPos('C',          'result')

        result, method_name = binarize(gray, method, thresh_val, block_size, C)

        cv2.putText(result, method_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        cv2.imshow('original', gray)
        cv2.imshow('result', result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captured.png', result)
            print(f'저장됨 - {method_name}')

    cap.release()
    cv2.destroyAllWindows()
