# Scale Alignment
# : 최소 제곱법(Least Squares)를 이용해 gt 이미지를 초기 뎁스 이미지와 같은 스케일로 변환

import numpy as np

def align_scale(gt, init_depth, mask=None):
    """
    GT depth를 초기 뎁스 스케일로 역정렬하기 위해 s, t를 추정.
    투명 물체 등 제외 영역은 mask로 제거.

    Args:
        gt: Ground Truth depth (H, W)
        init_depth: 초기 뎁스 (H, W)
        mask: 유효 픽셀 마스크 (1=유효, 0=제외)

    Returns:
        aligned_gt: 초기 뎁스 스케일로 변환된 GT
        s, t: 스케일, 오프셋
    """

    # Flatten
    # 스케일 정렬(s, t) 계산을 하기위해 gt, init_depth를 1차원 벡터로 펴기: (H, W) → (H × W,)
    gt_flat = gt.flatten()
    init_flat = init_depth.flatten()

    # Mask 처리
    if mask is not None:
        mask_flat = mask.flatten()
        
        # 뎁스 결손이 발생하지 않은 배경(Mask = 0)을 기준으로 정렬
        valid_idx = (mask_flat == 0) & np.isfinite(gt_flat) & np.isfinite(init_flat)
    else:
        valid_idx = np.isfinite(gt_flat) & np.isfinite(init_flat)

    # Mask 적용
    gt_valid = gt_flat[valid_idx]
    init_valid = init_flat[valid_idx]

    # 예외 처리(유효한 픽셀이 적은 경우)
    if len(gt_valid) < 10:
        raise ValueError("유효 픽셀이 너무 적습니다. (mask를 확인하세요)")

    # 최소제곱법으로 s, t 계산
    A = np.vstack([init_valid, np.ones(len(init_valid))]).T
    s, t = np.linalg.lstsq(A, gt_valid, rcond=None)[0]

    # GT를 초기 뎁스 스케일로 변환
    aligned_gt = (gt - t) / s

    return aligned_gt, s, t
