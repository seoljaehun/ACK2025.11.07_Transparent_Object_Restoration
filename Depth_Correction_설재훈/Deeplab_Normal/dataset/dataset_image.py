#  DataSet Custom 
# : Cleargrasp Dataset을 업로드 후 전처리하는 과정
#===========================================================================================
#       Type       file extension         Shape         Dtype    Channels
#===========================================================================================
#       RGB            .jpg          (1080, 1920, 3)    uint8       3 
#      Normal          .exr          (1080, 1920, 3)    float32     3
#===========================================================================================

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # OpenCV EXR 지원 활성화
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

#===========================
# Dataset Class
#===========================
class CleargaspDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
            데이터 셋 경로 설정 및 파일 리스트 초기화
        Args:
            root_dir (str): Cleargasp dataset root
            split (str): Train / Val / Test
            transform: optional augmentation (currently not applied)
        """
        # 인스턴스 변수 선언
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 데이터 별 경로를 인스턴트 변수로 선언
        self.rgb_dir = os.path.join(root_dir, "RGB_img", split)
        self.normal_dir = os.path.join(root_dir, "Normal_img", split)

        # RGB 이미지 파일 이름 목록을 저장하는 리스트
        self.files = sorted(os.listdir(self.rgb_dir))

        # 이미지들을 resize할 크기 지정
        self.resize_shape = (512, 288)  # (width, height)

    def __len__(self):
        """
            데이터 셋의 총 샘플 수 반환
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
            데이터 이미지 업로드 및 전처리 과정
        Args:
            idx (int): Dataset index Number
        """
        # 확장자를 뺀 파일 이름 받아오기
        filename = os.path.splitext(self.files[idx])[0]

        #===========================
        # Load RGB (.jpg)
        #===========================
        rgb_path = os.path.join(self.rgb_dir, filename + ".jpg") # 폴더 경로 + 파일이름 + 확장자
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)    # (H, W, 3), BGR 순서
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)      # BGR -> RGB 순서 변환
        rgb = rgb.astype(np.float32) / 255.0            # float32 적용 및 픽셀 값 정규화(0 ~ 1)

        #===========================
        # Load Surface Normal (.exr)
        #===========================
        normal_path = os.path.join(self.normal_dir, filename + ".exr")
        normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)  # float32, (H, W, 3)
        if normal_img is None:
            normal_img = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)

        #===========================
        # Resize All Images
        #===========================
        target_w, target_h = self.resize_shape

        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        normal_img = cv2.resize(normal_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        #===========================
        # Re-normalize a Normal Vector
        #===========================
        norm = np.linalg.norm(normal_img, axis=2, keepdims=True)
        epsilon = 1e-6
        normal_img[norm.squeeze() < epsilon] = 0    # norm 값이 0에 가까운 배경은 0으로 처리
        normal_img = normal_img / (norm + epsilon)
    
        #===========================
        # Convert to Tensor: (H, W, C) -> (C, H, W) 변환
        #===========================
        rgb = rgb.transpose(2, 0, 1)                          # (3, H, W)
        normal_img = normal_img.transpose(2, 0, 1)            # (3, H, W)

        sample = {
            # 입력(input): RGB 이미지(3차원)
            "input": torch.from_numpy(rgb).float(),  # (3, H, W)
            # 목표(target): Surface Normal 이미지(3차원)
            "target": torch.from_numpy(normal_img).float(),  # (3, H, W)
            "filename": filename
        }

        # 각 데이터 이미지 반환
        return sample
