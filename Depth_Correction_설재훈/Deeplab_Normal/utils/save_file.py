# Save file
# : Surface Normal 맵을 png 또는 exr 파일로 저장

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

def save_normal_as_png(normal_array, save_path):
    """
        Surface Normal 맵을 보기 쉽게 [0, 255]로 정규화 후 .png 이미지로 저장하는 함수
    Args:
        normal_array: 입력 Surface Normal 맵, float 형태의 (H, W, 3)
        save_path: 저장할 png 폴더 경로
    """
    
    # Normal 값의 범위를 [-1, 1] -> [0, 255]로 변환
    normal_to_save = ((normal_array * 0.5 + 0.5) * 255).astype(np.uint8)
    
    # OpenCV 저장을 위해 RGB -> BGR로 채널 순서 변경
    normal_to_save = cv2.cvtColor(normal_to_save, cv2.COLOR_RGB2BGR)
    
    # 특정 경로에 .png 확장자로 저장, uint8
    cv2.imwrite(save_path, normal_to_save)
    
def save_normal_as_exr(normal_array, save_path):
    """
        정규화를 하지 않고 실제 Surface Normal 값을 그대로 .exr 파일로 저장
    Args:
        normal_array: 입력 Surface Normal 맵, float 형태의 (H, W, 3)
        save_path: 저장할 exr 폴더 경로
    """
    # OpenCV 저장을 위해 RGB -> BGR로 채널 순서 변경
    normal_to_save = cv2.cvtColor(normal_array.astype(np.float32), cv2.COLOR_RGB2BGR)

    # 특정 경로에 .exr 확장자로 저장, float32
    cv2.imwrite(save_path, normal_to_save)
    