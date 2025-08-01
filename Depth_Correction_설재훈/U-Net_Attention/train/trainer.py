#  Train loop
# : one epoch 당 train, validate loop 정의, 
#  모델을 학습시키기 위해 데이터셋을 반복하여 가중치를 업데이트하는 과정

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

def save_exr(array, path):
    """
    attention weight를 exr로 저장 (float32 형식)
    Args:
        array: 저장할 데이터, torch.Tensor (B, C, H, W)
        path: 저장 경로
    """
    # 만약 array가 PyTorch의 텐서 형태라면 그래디언트 추적을 끊고 CPU로 이동 후 numpy로 변환
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    # (C, H, W) -> 평균 (H, W) 변환
    if array.ndim == 3:
        array = array.mean(axis=0)
        array = (array - array.min()) / (array.max() - array.min() + 1e-6)  # 0~1 시각적 정규화

    # 특정 경로에 .exr 확장자로 저장
    cv2.imwrite(path, array.astype(np.float32))
    
#==========================
# Train One Epoch
#==========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, max_grad_norm=5.0, save_target_filename=None, save_dir=None):
    """
        one epoch 동안 모델 학습을 수행
    Args:
        model: 학습할 네트워크(U-Net + Attention)
        loader: 학습 데이터 로더(train_Loader, val_Loader)
        optimizer: 옵티마이저(Adam)
        criterion: 손실 함수(Masked L1 Loss)
        device: 실행 디바이스(GPU)
        epoch: 현재 epoch 수
        max_grad_norm: gradient clipping 값 -> 학습 안정화(Loss 진동을 완화)
        save_target_filename: Attention weight Map을 저장할 이미지 이름
        save_dir: Attention weight Map을 저장할 경로
    Returns:
        train average Loss: train 한 epoch 동안 전체 데이터 셋의 평균 손실
    """
    # 학습 모드 활성화
    model.train()
    # 손실 누적 변수 초기화
    total_loss = 0
    
    # 배치 반복
    for batch in tqdm(loader, desc="Training", leave=False): 
        # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트, leave=False: epoch 끝나면 진행 바 지우기 
        # Dataset Loader에서 하나의 batch를 꺼내서 학습하는 반복문
        
        # 배치 데이터를 GPU로 이동
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        normal = batch["normal"].to(device)
        occlusion = batch["occlusion"].to(device)
        contact = batch["contact"].to(device)
        filenames = batch["filename"]

        # 입력의 4번째 채널 -> init_Depth 추출
        init_depth = inputs[:, 3:4, :, :]

        # Gradient 초기화
        optimizer.zero_grad()
        
        # 모델의 Forward 연산
        outputs = model(inputs, occ_edge=occlusion, contact_edge=contact, normal_img=normal)
        # 손실 계산
        loss = criterion(outputs, target, init_depth, mask, normal)  # Mask-L1 Loss
        # 손실 기반 Gradient 계산
        loss.backward()
        # Gradient Clipping -> Gradient 폭발 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # 계산된 Gradient로 모델 가중치 업데이트
        optimizer.step()
        # 배치별 Loss를 누적
        total_loss += loss.item()
        
        # 이미지 파일 이름을 리스트 형태로 변환
        filenames_list = list(filenames)
        
        # 특정 이미지의 Attention weight Map을 저장할 조건(= 이미지 이름)
        if save_target_filename in filenames_list:
            idx = filenames.index(save_target_filename) # 배치 index
            
            # 저장 경로 생성
            ep_name = f"epoch_{epoch + 1:02d}.exr"
            save_exr(model.occ_edge_att.att_weight[idx], os.path.join(save_dir, "occlusion", ep_name))
            save_exr(model.contact_edge_att.att_weight[idx], os.path.join(save_dir, "contact", ep_name))
            save_exr(model.normal_att.att_weight[idx], os.path.join(save_dir, "normal", ep_name))
            
    # 1 Epoch 당 평균 Loss 반환
    return total_loss / len(loader)


#==========================
# Validate One Epoch
#==========================
def validate_one_epoch(model, loader, criterion, device):
    """
        one epoch 동안 모델 평가를 수행
    Args:
        model: 학습할 네트워크(U-Net + Attention)
        loader: 학습 데이터 로더(train_Loader, val_Loader)
        criterion: 손실 함수(Masked L1 Loss)
        device: 실행 디바이스(GPU)
    Returns:
        validation average Loss: validation 한 epoch 동안 전체 데이터 셋의 평균 손실
    """
    # 평가 모드 활성화
    model.eval()
    # 손실 누적 변수 초기화
    total_loss = 0

    # Gradient 계산 비활성화, 검증 단계에서는 가중치 업데이트 안함
    with torch.no_grad():
        # 배치 반복
        for batch in tqdm(loader, desc="Validation", leave=False):
            # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트, leave=False: epoch 끝나면 진행 바 지우기 
            # Dataset Loader에서 하나의 batch를 꺼내서 평가하는 반복문
        
            # 배치 데이터를 GPU로 이동
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            normal = batch["normal"].to(device)
            occlusion = batch["occlusion"].to(device)
            contact = batch["contact"].to(device)

            # 입력의 4번째 채널 -> init_Depth 추출
            init_depth = inputs[:, 3:4, :, :]
        
            # 모델의 Forward 연산
            outputs = model(inputs, occ_edge=occlusion, contact_edge=contact, normal_img=normal)
            # 손실 계산
            loss = criterion(outputs, target, init_depth, mask, normal) # Mask-L1 Loss
            
            # 배치별 Loss를 누적
            total_loss += loss.item()

    # 1 Epoch 당 평균 Loss 반환
    return total_loss / len(loader)