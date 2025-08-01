#  Inference loop
# : 학습된 모델을 불러와 Test dataset에 대한 결과를 예측하고, 저장

import torch
from torch.utils.data import DataLoader
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # OpenCV EXR 지원 활성화
import cv2
import numpy as np
from config import Config
from model.Unet_Residual import UNetBaseline
from dataset.dataset_image import CleargaspDataset
from utils.checkpoint import load_checkpoint

#==========================
# Utility Functions
#==========================
def save_depth_image(depth_array, save_path, normalize=True):
    """
        Depth Map을 보기 쉽게 .png 이미지로 저장하는 함수
    Args:
        depth_array: 입력 Depth Map (float 형태)
        save_path: 저장할 png 파일 경로
        normalize: 최소, 최대값을 기준으로 정규화(0 ~ 255)
    """
    
    # 0 ~ 255 범위로 정규화
    if normalize:
        # 현재 뎁스 값의 최솟값과 최댓값을 찾음
        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)
        # 뎁스 범위를 0 ~ 1로 변환
        depth_vis = (depth_array - depth_min) / (depth_max - depth_min + 1e-8)
        # 뎁스 범위를 0 ~ 255로 정규화
        depth_vis = (depth_vis * 255).astype(np.uint8)
    else:
        # 이미 뎁스 범위가 0 ~ 1인 경우
        depth_vis = (depth_array * 255).astype(np.uint8)
        
    # 정규화된 Depth Map을 png로 저장
    cv2.imwrite(save_path, depth_vis)

def save_exr(depth_array, save_path):
    """
        정규화를 하지 않고 실제 뎁스 값을 그대로 .exr 파일로 저장
    Args:
        depth_array: 입력 Depth Map (float 형태)
        save_path: 저장할 exr 파일 경로
    """
    
    # 실제 Depth Map을 float32 값으로 exr로 저장
    cv2.imwrite(save_path, depth_array.astype(np.float32))

#==========================
# Inference Main
#==========================
if __name__ == "__main__":
    # config() 클래서에서 경로, 하이퍼 파라미터, 디바이스 설정 정보 불러오기
    cfg = Config()
    # device = GPU
    device = cfg.device

    # CleargaspDataset 클래스에서 test 데이터 셋 로드
    test_dataset = CleargaspDataset(root_dir=cfg.root_dir, split="Test")
    # DataLoader 초기화
    # batch_size: batch 수 만큼 데이터를 묶어 GPU에 전달 (= 1)
    # shuffle = False: 순서를 섞지 않고 testing
    # num_workers: 멀티스레드로 데이터 로딩 속도 향상
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # UNetAttention 클래스에서 모델 로드 -> GPU
    model = UNetBaseline(n_channels=cfg.in_channels, n_classes=cfg.out_channels).to(device)

    # Load Best Model
    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pth")
    load_checkpoint(model, optimizer=None, path=best_ckpt, device=device)
    
    # 평가 모드 활성화
    model.eval()

    # Output Directory (결과를 저장할 상위 폴더)
    output_dir = cfg.inference_output_dir
    subfolders = ["rgb", "init_depth", "pred_depth", "gt_depth", "exr_pred", "exr_gt"]
    
    # 저장할 폴더가 없는 경우 생성
    for folder in subfolders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        
    # Inference Loop
    # Gradient 계산 비활성화, test 단계에서는 가중치 업데이트 안함
    with torch.no_grad():
        # 배치 반복
        for batch in test_loader:
            # Dataset Loader에서 하나의 batch를 꺼내서 평가하는 반복문
            
            # 배치 데이터를 GPU로 이동
            inputs = batch["input"].to(device)
            mask = batch["mask"].to(device)
            filename = batch["filename"][0]

            # 입력의 4번째 채널 -> init_Depth 추출 후 numpy로 변환
            init_depth = inputs[:, 3:4, :, :].cpu().numpy()[0, 0]  # (H, W)
            # target의 1번째 채널 -> gt_Depth 추출 후 numpy로 변환
            gt_residual = batch["target"].cpu().numpy()[0, 0]      # (H, W)

            # Predict Residual
            outputs = model(inputs)
            outputs = outputs * mask
            
            # 예측 오차 Depth를 numpy로 변환
            pred_residual = outputs.cpu().numpy()[0, 0]

            # 복원된 Depth, gt Depth 계산
            pred_depth = init_depth + pred_residual
            gt_depth = init_depth + gt_residual

            # 입력의 1~3번째 채널 -> RGB 이미지 추출 후 초기 상태로 변환
            rgb = (inputs[:, :3, :, :].cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

            #==========================
            # Save Paths
            #==========================
            rgb_path = os.path.join(output_dir, "rgb", f"{filename}.png")
            init_depth_path = os.path.join(output_dir, "init_depth", f"{filename}.png")
            pred_depth_path = os.path.join(output_dir, "pred_depth", f"{filename}.png")
            gt_depth_path = os.path.join(output_dir, "gt_depth", f"{filename}.png")

            pred_exr_path = os.path.join(output_dir, "exr_pred", f"{filename}.exr")
            gt_exr_path = os.path.join(output_dir, "exr_gt", f"{filename}.exr")

            #==========================
            # Save Visualization PNG
            #==========================
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            save_depth_image(init_depth, init_depth_path)
            save_depth_image(pred_depth, pred_depth_path)
            save_depth_image(gt_depth, gt_depth_path)

            #==========================
            # Save Actual Depth EXR
            #==========================
            save_exr(pred_depth, pred_exr_path)
            save_exr(gt_depth, gt_exr_path)

            print(f"[INFO] Saved: {filename}")
