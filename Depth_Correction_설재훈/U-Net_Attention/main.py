#  Main loop
# : 설정한 epoch만큼 train과 validation을 반복하여 최적 모델을 저장

import torch
torch.cuda.empty_cache()  # GPU 캐시 초기화
from torch.utils.data import DataLoader
import os
from config import Config
from model.Unet_attention import UNetAttention
from loss.loss1_basic import MaskedL1Loss
from dataset.dataset_image import CleargaspDataset
from train.trainer import train_one_epoch, validate_one_epoch
from utils.checkpoint import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau  
    
if __name__ == "__main__":
    # config() 클래서에서 경로, 하이퍼 파라미터, 디바이스 설정 정보 불러오기
    cfg = Config()
    # device = GPU
    device = cfg.device

    #==========================
    # Dataset & DataLoader
    #==========================
    
    # CleargaspDataset 클래스에서 train, val 데이터 셋 로드
    train_dataset = CleargaspDataset(root_dir=cfg.root_dir, split="Train")
    val_dataset = CleargaspDataset(root_dir=cfg.root_dir, split="Val")

    # DataLoader 초기화
    # batch_size: batch 수 만큼 데이터를 묶어 GPU에 전달
    # shuffle = True: 매 epoch마다 순서를 섞어 학습 안정화
    # num_workers: 멀티스레드로 데이터 로딩 속도 향상
    # pin_memory=True: GPU로 데이터를 옮길 때 속도 향상
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 검증에서는 shuffle = False (순서 안중요함)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    #==========================
    # Model, Loss, Optimizer
    #==========================
    
    # UNetAttention 클래스에서 모델 로드 -> GPU
    model = UNetAttention(n_channels=cfg.in_channels, n_classes=cfg.out_channels).to(device)
    # MaskedL1Loss 클래스에서 Loss 함수 로드 -> GPU
    criterion = MaskedL1Loss()
    # optimizer 설정 -> Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate) # 여기서 learning_rate는 초기값
    
    # Schedular 추가: Val Loss를 기반으로 learning_rate 조절
    # mode='min': Val Loss가 줄어야 개선으로 판단
    # factor=0.5: LR을 절반으로 줄임
    # patience=3: 3 epoch 동안 개선이 없으면 LR 감소
    # threshold: 감소폭 기준(이 값보다 작으면 개선 없다고 인식)
    # threshold_mode='rel': 상대적인 변화율 기준, 현재 best_loss 대비 몇 %나 줄었는지를 기준
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, threshold_mode='rel')

    #==========================
    # Checkpoint 설정
    #==========================
    
    # check point 폴더 생성(없으면 생성, 있으면 무시)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # 초기 변수 설정
    best_val_loss = float("inf") # 초기 Loss = 무한대
    start_epoch = 0
    resume = True   # 중단한 학습을 이어서 할지 여부

    # 중단한 학습을 이어서 하는 경우
    if resume:
        # 마지막 학습 상태가 저장된 checkpoint 파일 경로
        last_ckpt = os.path.join(cfg.checkpoint_dir, "last.pth")
        
        # 마지막 모델의 가중치, optimizer, epoch, loss 등을 불러옴
        if os.path.exists(last_ckpt):
            completed_epoch, best_val_loss = load_checkpoint(model, optimizer, last_ckpt, device)
            start_epoch = completed_epoch + 1
            print(f"[INFO] Resuming training from epoch {start_epoch + 1}")

    #==========================
    # Save Attention Weight Map
    #==========================

    # Attention weight 이미지 이름
    target_filename = "cup-with-waves-train (1)"

    # 저장할 attention weight 폴더
    attention_base_dir = cfg.attention_weight_dir

    # 저장할 각 경로에 폴더 생성
    os.makedirs(os.path.join(attention_base_dir, "occlusion"), exist_ok=True)
    os.makedirs(os.path.join(attention_base_dir, "contact"), exist_ok=True)
    os.makedirs(os.path.join(attention_base_dir, "normal"), exist_ok=True)
    
    #==========================
    # Training & Validation Loop
    #==========================
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch [{epoch + 1}/{cfg.epochs}]")

        # 1. Training
        avg_train_loss, train_l1, train_grad, train_norm = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, save_target_filename=target_filename, save_dir=attention_base_dir)
        
        # 2. Validation
        avg_val_loss, val_l1, val_grad, val_norm = validate_one_epoch(model, val_loader, criterion, device)

        # Scheduler 적용
        scheduler.step(avg_val_loss)
        # 현재 LR 확인
        current_lr = optimizer.param_groups[0]['lr']
        
        # 평균 Loss 값 출력
        print(f"[Epoch {epoch + 1}] "
              f"Train Loss: {avg_train_loss:.4f} (L1: {train_l1:.4f}, Grad: {train_grad:.4f}, Norm: {train_norm:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (L1: {val_l1:.4f}, Grad: {val_grad:.4f}, Norm: {val_norm:.4f}) | "
              f"LR: {current_lr:.6f}")

        #==========================
        # Checkpoint 저장
        #==========================
        
        # 항상 현재 모델 상태 저장 (파일: last.pth)
        save_checkpoint(model, optimizer, epoch, avg_val_loss, os.path.join(cfg.checkpoint_dir, "last.pth"))
        
        # Validation Loss가 개선되면 Best 모델 갱신 후 상태 저장 (파일: best.pth)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, os.path.join(cfg.checkpoint_dir, "best.pth"))
            print(">>> Best model updated!")
