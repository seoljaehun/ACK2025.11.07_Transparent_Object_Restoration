# Inference loop
# : 학습된 모델을 불러와 Test dataset에 대한 결과를 예측하고, 저장

import torch
from torch.utils.data import DataLoader
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # OpenCV EXR 지원 활성화
import cv2
import numpy as np
from tqdm import tqdm
from config import Config
from dataset.dataset_image import CleargaspDataset
from model.Deeplab_Normal import DeepLabv3Plus_for_Normals
from utils.checkpoint import load_checkpoint
from utils.save_file import save_normal_as_png, save_normal_as_exr

# ==========================
# Inference Main
# ==========================
if __name__ == "__main__":
    # config() 클래서에서 경로, 하이퍼 파라미터, 디바이스 설정 정보 불러오기
    cfg = Config()
    # device = GPU
    device = cfg.device

    # ==========================
    # Dataset & DataLoader
    # ==========================
    # CleargaspDataset 클래스에서 test 데이터 셋 로드
    test_dataset = CleargaspDataset(root_dir=cfg.root_dir, split="Test")
    # DataLoader 초기화
    # batch_size: batch 수 만큼 데이터를 묶어 GPU에 전달 (= 1)
    # shuffle = False: 순서를 섞지 않고 testing
    # num_workers: 멀티스레드로 데이터 로딩 속도 향상
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg.num_workers
    )

    # ==========================
    # Model Load
    # ==========================
    # DeepLabv3Plus_for_Normals 클래스에서 모델 로드 -> GPU
    model = DeepLabv3Plus_for_Normals().to(device)

    # Load Best Model
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")
    load_checkpoint(model, optimizer=None, path=best_ckpt_path, device=device)
    
    # 평가 모드 활성화
    model.eval()

    # ==========================
    # Output Directory
    # ==========================
    # 결과를 저장할 상위 폴더
    output_dir = cfg.inference_output_dir
    # 저장할 하위 폴더 리스트
    subfolders = ["rgb", "pred_normal_png", "gt_normal_png", "pred_normal_exr", "gt_normal_exr"]
    
    # 저장할 폴더가 없는 경우 생성
    for folder in subfolders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        
    # ==========================
    # Inference Loop
    # ==========================
    # Gradient 계산 비활성화, test 단계에서는 가중치 업데이트 안함
    with torch.no_grad():
        # tqdm으로 진행률 표시
        loop = tqdm(test_loader, leave=True, desc="Inferencing")
        # 배치 반복
        for data in loop:
            # Dataset Loader에서 하나의 batch를 꺼내서 평가하는 반복문
            
            # 배치 데이터를 GPU로 이동
            inputs = data["input"].to(device)
            targets = data["target"].to(device)
            filename = data["filename"][0]

            # 모델 Forward 연산
            predictions = model(inputs)
            
            # 텐서를 저장 가능한 numpy 배열로 변환
            # (1, C, H, W) -> (C, H, W) -> (H, W, C)
            rgb = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_normal = predictions.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_normal = targets.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # ==========================
            # Save Paths
            # ==========================
            rgb_path = os.path.join(output_dir, "rgb", f"{filename}.png")
            pred_png_path = os.path.join(output_dir, "pred_normal_png", f"{filename}.png")
            gt_png_path = os.path.join(output_dir, "gt_normal_png", f"{filename}.png")
            pred_exr_path = os.path.join(output_dir, "pred_normal_exr", f"{filename}.exr")
            gt_exr_path = os.path.join(output_dir, "gt_normal_exr", f"{filename}.exr")

            # ==========================
            # Save Visualization PNG
            # ==========================
            # RGB 이미지는 0~1 범위이므로 255를 곱해 uint8로 변환 후 BGR로 저장
            cv2.imwrite(rgb_path, cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            save_normal_as_png(pred_normal, pred_png_path)
            save_normal_as_png(gt_normal, gt_png_path)

            # ==========================
            # Save Actual Normal EXR
            # ==========================
            save_normal_as_exr(pred_normal, pred_exr_path)
            save_normal_as_exr(gt_normal, gt_exr_path)

    print(f"\n[INFO] Inference finished. Results saved to {output_dir}")