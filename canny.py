import cv2

"""
Canny 엣지 검출 모듈

[ canny_low ]  : 경계 후보 임계값. 낮을수록 끊긴 경계를 이어줌.
[ canny_high ] : 확실한 경계 임계값. 높을수록 뚜렷한 경계만 남음.
권장 비율 → canny_high = canny_low * 2~3  (예: low=50, high=150)
"""

def canny_edge(binary, low, high):
    if high <= low:
        high = low + 1
    return cv2.Canny(binary, low, high)
