# Scale Alignment
# : 최소 제곱법(Least Squares)를 이용해 예측 뎁스를 gt 뎁스와 같은 스케일로 변환

import torch
import numpy as np

def align_scale(gt, pred_depth):
    """
    예측 depth를 GT depth 스케일로 정렬하기 위해 s, t 값을 반환

    Args:
        gt: Ground Truth depth (B, 1, H, W)
        pred_depth: Prediction depth (B, 1, H, W)

    Returns:
        s, t: Scale and Shift parameters
    """

    # s와 t를 저장할 리스트
    s_batch = []
    t_batch = []
    
    # PyTorch Tensor -> NumPy Array 변환
    gt_np_batch = gt.detach().cpu().numpy()
    pred_np_batch = pred_depth.detach().cpu().numpy()
    
    batch_size = gt.shape[0]
    
    # 각 배치 아이템에 대해 개별적으로 s, t 계산, (B, 1, H, W) -> (1, H, W)
    for i in range(batch_size):
        
        # 현재 이미지 선택 (1, H, W) -> (H, W)
        gt_single = gt_np_batch[i, 0]
        pred_single = pred_np_batch[i, 0]
        
        # Flatten
        # 스케일 정렬(s, t) 계산을 하기위해 gt, pred_depth를 1차원 벡터로 펴기: (H, W) → (H × W,)
        gt_flat = gt_single.flatten()
        pred_flat = pred_single.flatten()
        
        # 최소제곱법으로 s, t 계산
        # gt = s * pred + t 관계 찾기
        A = np.vstack([pred_flat, np.ones(len(pred_flat))]).T
        s, t = np.linalg.lstsq(A, gt_flat, rcond=None)[0]
    
        # 리스트에 추가
        s_batch.append(s)
        t_batch.append(t)

    # s와 t를 PyTorch 텐서로 변환하여 반환
    s_tensor = torch.tensor(s_batch, device=gt.device, dtype=gt.dtype)
    t_tensor = torch.tensor(t_batch, device=gt.device, dtype=gt.dtype)
    
    return s_tensor, t_tensor
