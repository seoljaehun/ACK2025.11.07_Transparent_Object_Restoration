#  Check Point
# : 학습 중 모델의 last, best 가중치를 저장하고, 추론할 때 불러오는 역할

import torch

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """
        학습 중 모델의 가중치, optimizer, epoch, Loss를 저장
    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        epoch: int
        val_loss: float
        path: str, save path
    """
    # 학습 정보를 변수에 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),     # 모델의 모든 가중치
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,  # optimizer의 상태(momentum, learning rate 등)
        "epoch": epoch, # 저장 시점의 학습 epoch 수
        "val_loss": val_loss  # 현재까지의 Validation Loss 최소값
    }
    # checkpoint를 지정된 경로에 바이너리 형태로 저장
    torch.save(checkpoint, path)    
    print(f"[INFO] Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    학습 재개 또는 추론 시 학습된 모델의 가중치, 파라미터를 불러옴
    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer or None
        path: str, checkpoint file path
        device: torch.device
    Returns:
        epoch: int (last trained epoch)
        val_loss: float
    """
    # 저장된 check point 파일 읽기
    checkpoint = torch.load(path, map_location=device)
    
    # 모델의 가중치(state_dict) 복원
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer 상태(momentum, learning rate 등) 복원
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 마지막 학습 epoch 복원
    epoch = checkpoint.get("epoch", 0)
    # 현재까지의 Validation Loss 최소값 복원
    val_loss = checkpoint.get("val_loss", float("inf"))
    
    # 로드 정보 출력 후 epoch, val_loss 값만 반환
    # 가중치와 optimizer는 이미 model과 optimizer 객체에 로드된 상태
    print(f"[INFO] Checkpoint loaded from {path} | Epoch: {epoch+1} | Best Val Loss: {val_loss:.4f}")
    return epoch, val_loss
