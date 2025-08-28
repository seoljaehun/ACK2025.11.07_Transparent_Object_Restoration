#  Loss 함수
# : 딥러닝 모델의 Loss 함수 정의, cosine similarity loss

import torch
import torch.nn.functional as F

#==========================
# 1. cosine similarity loss
#==========================
def cosine_similarity_loss(pred_normals, gt_normals):
    """
        1. cosine similarity loss: 예측된 노멀 맵과 실제 노멀 맵 사이의 코사인 유사도 손실을 계산합니다.

    Args:
        pred_normals (torch.Tensor): 모델이 예측한 노멀 맵 (B, 3, H, W)
        gt_normals (torch.Tensor): 실제 정답 노멀 맵 (B, 3, H, W)

    Returns:
        torch.Tensor: 스칼라 형태의 최종 손실 값
    """
    # 1. 마스크 생성: 정답(gt) 노멀 벡터의 크기가 0보다 큰 유효한 픽셀만 True로 설정
    # 각 채널을 제곱해서 더한 값이 0보다 큰지 확인하여 유효한 벡터만 남김
    epsilon = 1e-6
    mask = torch.sum(gt_normals**2, dim=1) > epsilon

    # 2. 각 픽셀별로 코사인 유사도를 계산
    # 결과는 (B, H, W) 모양의 텐서, 값의 범위 [-1, 1]
    similarity = F.cosine_similarity(pred_normals, gt_normals, dim=1)

    # 3. 유사도를 손실로 변환
    # 유사도가 1(완벽히 일치)이면 loss는 0, -1(완전 반대)이면 loss는 2
    loss = 1 - similarity

    # 4. 마스크를 적용하여 유효한 픽셀의 손실 값만 남김
    masked_loss = loss[mask]
    
    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=pred_normals.device)
    
    return masked_loss.mean()
