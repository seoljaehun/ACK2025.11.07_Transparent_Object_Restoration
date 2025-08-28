#  Inference loop
# : 학습된 모델을 불러와 Test dataset에 대한 결과를 예측하고, 저장

import torch
from torch.utils.data import DataLoader
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # OpenCV EXR 지원 활성화
import cv2
import numpy as np
from config import Config
from model.Unet_attention import UNetAttention
from dataset.dataset_image import CleargaspDataset
from utils.checkpoint import load_checkpoint
from utils.save_file import save_image, save_exr

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
    model = UNetAttention(n_channels=cfg.in_channels, n_classes=cfg.out_channels).to(device)

    # Load Best Model
    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pth")
    load_checkpoint(model, optimizer=None, path=best_ckpt, device=device)
    
    # 평가 모드 활성화
    model.eval()

    # Output Directory (결과를 저장할 상위 폴더)
    output_dir = cfg.inference_output_dir
    subfolders = ["rgb", "init_depth", "pred_depth", "gt_depth", "exr_pred", "exr_gt", "exr_residual"]
    
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
            target = batch["target"].to(device)
            rgb = batch["rgb"].to(device)
            init = batch["init"].to(device)
            mask = batch["mask"].to(device)
            normal = batch["normal"].to(device)
            occlusion = batch["occlusion"].to(device)
            contact = batch["contact"].to(device)
            filename = batch["filename"][0]

            # 초기 상태로 변환
            gt_residual = target.cpu().numpy()[0, 0]    # (H, W)
            init_depth = init.cpu().numpy()[0, 0]  # (H, W)
            rgb = (rgb.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Predict Residual
            outputs = model(inputs, occ_edge=occlusion, contact_edge=contact, normal_img=normal)
            outputs = outputs * mask
            
            # 예측 오차 Depth를 numpy로 변환
            pred_residual = outputs.cpu().numpy()[0, 0]

            # 복원된 Depth, gt Depth 계산
            pred_depth = init_depth + pred_residual
            gt_depth = init_depth + gt_residual

            #==========================
            # Save Paths
            #==========================
            rgb_path = os.path.join(output_dir, "rgb", f"{filename}.png")
            init_depth_path = os.path.join(output_dir, "init_depth", f"{filename}.png")
            pred_depth_path = os.path.join(output_dir, "pred_depth", f"{filename}.png")
            gt_depth_path = os.path.join(output_dir, "gt_depth", f"{filename}.png")

            pred_exr_path = os.path.join(output_dir, "exr_pred", f"{filename}.exr")
            gt_exr_path = os.path.join(output_dir, "exr_gt", f"{filename}.exr")
            residual_exr_path = os.path.join(output_dir, "exr_residual", f"{filename}.exr")

            #==========================
            # Save Visualization PNG
            #==========================
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            save_image(init_depth, init_depth_path)
            save_image(pred_depth, pred_depth_path)
            save_image(gt_depth, gt_depth_path)

            #==========================
            # Save Actual Depth EXR
            #==========================
            save_exr(pred_depth, pred_exr_path)
            save_exr(gt_depth, gt_exr_path)
            save_exr(pred_residual, residual_exr_path)

            print(f"[INFO] Saved: {filename}")
