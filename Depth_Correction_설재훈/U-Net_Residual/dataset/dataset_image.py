#  DataSet Custom 
# : Cleargrasp Dataset을 업로드 후 전처리하는 과정
#===========================================================================================
#       Type       file extension         Shape         Dtype    Channels
#===========================================================================================
#       RGB            .jpg          (1080, 1920, 3)    uint8       3   
#    init_Depth        .pfm          (1080, 1920)       float32     1
#     gt_Depth         .exr          (1080, 1920, 3)    float32     3
#      Normal          .exr          (1080, 1920, 3)    float32     3
#       Mask           .png          (1080, 1920)       uint8       1
#===========================================================================================

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # OpenCV EXR 지원 활성화
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from utils.scale_alignment import align_scale  # utils -> scale_alignment -> align_scale 함수 불러오기

#===========================
# PFM Loader
#===========================
def load_pfm(file_path):
    """Load PFM depth file as numpy array"""
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        color = header == 'PF'
        dim_line = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode('utf-8').rstrip())
        data = np.fromfile(f, '<f')  # little-endian float
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM stores images upside down
        return data

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
        self.init_depth_dir = os.path.join(root_dir, "Depth_img", split)
        self.gt_depth_dir = os.path.join(root_dir, "GT_img", split)
        self.mask_dir = os.path.join(root_dir, "Mask_img", split)
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
        # Load Init Depth (.pfm)
        #===========================
        init_depth_path = os.path.join(self.init_depth_dir, filename + ".pfm")
        init_depth = load_pfm(init_depth_path).astype(np.float32)   # float32, (H, W)
        
        # 정규화: Robust Min-Max
        # 극단값(outlier)를 제거하기위해 min(하위 1% 값), max(상위 1% 값)으로 설정
        lower_percentile = 1.0
        upper_percentile = 99.0
        init_min = np.percentile(init_depth, lower_percentile)
        init_max = np.percentile(init_depth, upper_percentile)
        scale = init_max - init_min
        # 1% 보다 작은 값들은 정규화시 0보다 작은 값, 99% 보다 큰 값들은 정규화시 1보다 큰 값으로 나옴
        # -> 0 ~ 1 사이로 값을 자르는 후처리 진행
        init_norm = np.clip((init_depth - init_min) / (scale + 1e-4), 0.0, 1.0)
        
        #===========================
        # Load GT Depth (.exr)
        #===========================
        gt_depth_path = os.path.join(self.gt_depth_dir, filename + ".exr")
        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)  # float32, (H, W, 3)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth[:, :, 0] # (H, W, 3) -> (H, W) shape 변환

        #===========================
        # Load Mask (PNG)
        #===========================
        mask_path = os.path.join(self.mask_dir, filename + ".png")
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
        mask = (mask_raw > 128).astype(np.uint8)  # 픽셀 이진분류: 1 = valid pixel, 0 = background

        #===========================
        # Align GT scale using mask
        #===========================
        
        # gt 이미지를 init_Depth 이미지의 스케일과 오프셋으로 정렬
        aligned_gt, s, t = align_scale(gt_depth, init_norm, mask) # (H, W)

        #===========================
        # Compute Residual Depth
        #===========================
        
        # 정답 데이터: gt 뎁스 맵 - 초기 뎁스 맵 = 오차 뎁스 맵
        residual = aligned_gt - init_norm  # (H, W)

        #===========================
        # Load Extra Loss Inputs
        #===========================
        normal_path = os.path.join(self.normal_dir, filename + ".exr")
        normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)  # float32, (H, W, 3)
        if normal_img is None:
            normal_img = np.zeros((init_norm.shape[0], init_norm.shape[1], 3), dtype=np.float32)

        #===========================
        # Resize All Images
        #===========================
        target_w, target_h = self.resize_shape

        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        init_norm = cv2.resize(init_norm, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        residual = cv2.resize(residual, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        normal_img = cv2.resize(normal_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        #===========================
        # Convert to Tensor: (H, W, C) -> (C, H, W) 변환
        #===========================
        rgb = rgb.transpose(2, 0, 1)                          # (3, H, W)
        init_norm = np.expand_dims(init_norm, axis=0)         # (1, H, W)
        residual = np.expand_dims(residual, axis=0)           # (1, H, W)
        normal_img = normal_img.transpose(2, 0, 1)            # (3, H, W)
        mask = np.expand_dims(mask, axis=0)                   # (1, H, W)

        sample = {
            # 입력(input): RGB 이미지(3차원) + init_Depth_norm 이미지(1차원) = 4차원
            "input": torch.from_numpy(np.concatenate([rgb, init_norm], axis=0)).float(),  # (4, H, W)
            # 목표(target): residual 이미지(1차원)
            "target": torch.from_numpy(residual).float(),  # (1, H, W)
            "rgb": torch.from_numpy(rgb).float(),  # (3, H, W), 정규화된 상태(0~1)
            "normal": torch.from_numpy(normal_img).float(),
            "mask": torch.from_numpy(mask).float(),
            "filename": filename
        }

        # 각 데이터 이미지 반환
        return sample
