# Save file
# : Depth, RGB, Attention 등을 png 또는 exr 파일로 저장

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

def save_image(depth_array, save_path):
    """
        Depth Map을 보기 쉽게 [0, 255]로 정규화 후 .png 이미지로 저장하는 함수
    Args:
        depth_array: 입력 Depth Map (float 형태)
        save_path: 저장할 png 폴더 경로
    """
    
    # 정규화
    # 현재 뎁스 값의 최솟값과 최댓값을 찾음
    depth_min = np.min(depth_array)
    depth_max = np.max(depth_array)
    # 뎁스 범위를 0 ~ 1로 변환
    depth_norm = (depth_array - depth_min) / (depth_max - depth_min + 1e-8)
    # 뎁스 범위를 0 ~ 255로 정규화
    depth_norm = (depth_norm * 255).astype(np.uint8)
        
    # 특정 경로에 .png 확장자로 저장, unit8
    cv2.imwrite(save_path, depth_norm)
    
def save_exr(depth_array, save_path):
    """
        정규화를 하지 않고 실제 뎁스 값을 그대로 .exr 파일로 저장
    Args:
        depth_array: 입력 Depth Map (float 형태)
        save_path: 저장할 exr 폴더 경로
    """
    
    # 특정 경로에 .exr 확장자로 저장, float32
    cv2.imwrite(save_path, depth_array.astype(np.float32))