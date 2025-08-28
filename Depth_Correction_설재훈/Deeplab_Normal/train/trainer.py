#  Train loop
# : one epoch 당 train, validate loop 정의, 
#  모델을 학습시키기 위해 데이터셋을 반복하여 가중치를 업데이트하는 과정

import torch
from tqdm import tqdm

#==========================
# Train One Epoch
#==========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
        one epoch 동안 모델 학습을 수행

    Args:
        model: 학습할 네트워크(DeepLabV3+)
        loader: 학습 데이터 로더(train_Loader, val_Loader)
        optimizer: 옵티마이저(Adam)
        criterion: 손실 함수(cosine similarity loss)
        device: 실행 디바이스(GPU)

    Returns:
        train average Loss: 해당 Train 에포크의 평균 학습 손실 값
    """
    
    # 학습 모드 활성화
    model.train()
    
    # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트, leave=True: epoch 끝나도 진행 바 유지
    loop = tqdm(loader, leave=True, desc="Training")
    
    # 손실 누적 변수 초기화
    total_loss = 0

    for data in loop:
        inputs = data['input'].to(device)
        targets = data['target'].to(device)

        # 순전파 (Forward pass)
        # 모델의 Forward 연산
        predictions = model(inputs)
        # 손실 계산
        loss = criterion(predictions, targets)

        # 역전파 (Backward pass)
        # Gradient 초기화
        optimizer.zero_grad()
        # 손실 기반 Gradient 계산
        loss.backward()
        
        # 계산된 Gradient로 모델 가중치 업데이트
        optimizer.step()

        # 배치별 Loss를 누적
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # 1 Epoch 당 평균 Loss 반환
    return total_loss / len(loader)

#==========================
# Validate One Epoch
#==========================
def validate(model, loader, criterion, device):
    """
        one epoch 동안 모델 평가를 수행

    Args:
        model: 학습할 네트워크(DeepLabV3+)
        loader: 학습 데이터 로더(train_Loader, val_Loader)
        criterion: 손실 함수(cosine similarity loss)
        device: 실행 디바이스(GPU)

    Returns:
        validation average Loss: 해당 Validation 에포크의 평균 검증 손실 값
    """
    
    # 평가 모드 활성화
    model.eval()
    
    # 손실 누적 변수 초기화
    total_loss = 0
    
    # tqbm: 진행 상태를 표시하는 라이브러리, desc: 진행 바 앞에 표시할 텍스트, leave=True: epoch 끝나도 진행 바 유지
    loop = tqdm(loader, leave=True, desc="Validation")
    
    # Gradient 계산 비활성화, 검증 단계에서는 가중치 업데이트 안함
    with torch.no_grad():
        for data in loop:
            inputs = data['input'].to(device)
            targets = data['target'].to(device)
            
            # 모델의 Forward 연산
            predictions = model(inputs)
            # 손실 계산
            loss = criterion(predictions, targets)
            
            # 배치별 Loss를 누적
            total_loss += loss.item()
            
    # 1 Epoch 당 평균 Loss 반환
    return total_loss / len(loader)