#  Main loop
# : 설정한 epoch만큼 train과 validation을 반복하여 최적 모델을 저장

import torch
torch.cuda.empty_cache()  # GPU 캐시 초기화
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config
from dataset.dataset_image import CleargaspDataset
from model.Deeplab_Normal import DeepLabv3Plus_for_Normals
from loss.loss import cosine_similarity_loss
from utils.checkpoint import save_checkpoint, load_checkpoint
from train.trainer import train_one_epoch, validate

if __name__ == "__main__":
    # 설정값 로드 및 폴더 생성
    cfg = Config()
    device = cfg.device
    
    # check point 폴더 생성(없으면 생성, 있으면 무시)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

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
    
    # DeepLabv3Plus_for_Normals 클래스에서 모델 로드 -> GPU
    model = DeepLabv3Plus_for_Normals().to(device)
    # cosine_similarity_loss 함수 로드 -> GPU
    criterion = cosine_similarity_loss
    # optimizer 설정 -> Adam
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Val Loss가 개선되지 않으면 LR을 동적으로 감소시키는 스케줄러
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    #==========================
    # Checkpoint 설정
    #==========================
    best_val_loss = float("inf")
    start_epoch = 0
    resume = True  # True로 설정하면 마지막 학습부터 이어서 시작

    if resume:
        last_ckpt_path = os.path.join(cfg.checkpoint_dir, "last.pth")
        if os.path.exists(last_ckpt_path):
            print(f"[INFO] Resuming training from {last_ckpt_path}")
            completed_epoch, best_val_loss = load_checkpoint(model, optimizer, last_ckpt_path, device)
            start_epoch = completed_epoch + 1
            print(f"       Resumed from epoch {start_epoch+1}. Best validation loss so far: {best_val_loss:.4f}")

    #==========================
    # Training & Validation Loop 
    #==========================
    for epoch in range(start_epoch, cfg.epochs):
        
        # 1. Training
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2. Validation
        avg_val_loss = validate(model, val_loader, criterion, device)

        # 3. Scheduler 업데이트
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 4. 로그 출력
        print(f"Epoch [{epoch + 1}/{cfg.epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        #==========================
        # Checkpoint 저장
        #==========================
        
        # 항상 현재 모델 상태 저장 (파일: last.pth)
        save_checkpoint(model, optimizer, epoch, avg_val_loss, os.path.join(cfg.checkpoint_dir, "last.pth"))
        
        # Validation Loss가 개선되면 Best 모델 갱신 후 상태 저장 (파일: best.pth)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f">>> Best model updated! New best validation loss: {best_val_loss:.4f}")
            save_checkpoint(model, None, epoch, best_val_loss, os.path.join(cfg.checkpoint_dir, "best.pth"))
