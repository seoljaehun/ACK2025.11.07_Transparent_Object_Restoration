#  Mask-L1 Loss 함수
# : 딥러닝 모델의 Loss 함수 정의, Masked Smooth L1 Loss + Edge-aware Smoothness Loss

import torch
import torch.nn as nn

# Loss 정의
class MaskedL1Loss(nn.Module):
    """
        Masked Smooth L1 Loss + Edge-aware Smoothness Loss
        1. Masked Smooth L1 Loss: 예측된 Depth Map의 오차 예측 정확도를 높이기 위한 Loss
        2. Edge-aware Smoothness Loss: 예측된 Depth Map의 거칠기를 해결하고, 공간적 일관성을 높이기 위한 Loss
    """
    def __init__(self, beta=1.0, lambda_smooth=0.05): 
        # 부모 클래스(nn.Module) 초기화
        super(MaskedL1Loss, self).__init__()
        self.beta = beta    # beta: Smooth L1의 Huber Loss 전환점
        self.lambda_smooth = lambda_smooth # Edge-aware smoothness loss의 가중치

    def gradient_x(self, img):
        """
            x축 방향 gradient 계산 (가로 방향 픽셀 차이)
            img: (B, C, H, W)
            return: (B, C, H, W-1)
        """
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(self, img):
        """
            y축 방향 gradient 계산 (세로 방향 픽셀 차이)
            img: (B, C, H, W)
            return: (B, C, H-1, W)
        """
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    def forward(self, pred, target, mask, rgb):
        """
            총 손실 값 계산
        Args:
            pred: 모델 예측값, (B, 1, H, W)
            target: gt 값, (B, 1, H, W)
            mask: 투명 영역 segmentation mask (1 = valid, 0 = invalid), (B, 1, H, W)
            rgb: 입력 RGB 이미지 (B, 3, H, W)
        """
        #==========================
        # 1. Masked Smooth L1 Loss
        #==========================
        
        # 예측값과 gt 값의 절대 차이 계산, 차이가 너무 큰 값은 20으로 제한하여 안정성 확보
        diff = torch.clamp(torch.abs(pred - target), max=20.0)
        
        # Smooth L1 Loss공식 적용
        # diff < beta -> 작은 오차인 경우 L2 Loss: {0.5 * (y - y')^2}/beta
        # diff > beta -> 큰 오차인 경우 L1 Loss: {(y - y') - 0.5 * beta}
        smooth_L1 = torch.where(diff < self.beta,
                           0.5 * (diff ** 2) / self.beta,
                           diff - 0.5 * self.beta)
        
        # 마스크 적용: 투명 물체 영역만 Loss 계산 (배경 영역의 Loss 값은 0으로 무시)
        masked_loss = smooth_L1 * mask
        
        # 모든 픽셀의 Loss 평균 값 
        masked_loss_mean = masked_loss.sum() / (mask.sum() + 1e-8)
        
        #==========================
        # 2. Edge-aware Smoothness Loss
        #==========================
        
        # 예측된 Depth Map의 Gradient(x, y축)
        pred_grad_x = self.gradient_x(pred)
        pred_grad_y = self.gradient_y(pred)

        # 입력 RGB 이미지의 Gradient(x, y축) -> Edge weight로 사용
        img_grad_x = torch.mean(torch.abs(self.gradient_x(rgb)), 1, keepdim=True)
        img_grad_y = torch.mean(torch.abs(self.gradient_y(rgb)), 1, keepdim=True)

        # Edge-aware weight: 경계에서는 smoothing 약하게, 평탄 영역에서는 강하게
        # 이미지 gradient가 클수록 weight 작아짐 → 경계에서 smoothness 약함
        # 이미지 gradient가 작을수록 weight 큼 → 평탄 영역에서 smoothness 강함
        weight_x = torch.exp(-img_grad_x)
        weight_y = torch.exp(-img_grad_y)

        # mask는 gradient 크기에 맞게 슬라이스 (W-1, H-1 차이 보정)
        mask_x = mask[:, :, :, :-1]  # x방향 크기 맞추기
        mask_y = mask[:, :, :-1, :]  # y방향 크기 맞추기

        # x, y방향의 Smooth Loss Map 계산
        smooth_map_x = torch.abs(pred_grad_x) * weight_x
        smooth_map_y = torch.abs(pred_grad_y) * weight_y

        # Smoothness Loss 계산 (배경 영역의 Loss 값은 0으로 무시)
        smoothness_loss = (smooth_map_x * mask_x).sum() / (mask_x.sum() + 1e-8) + (smooth_map_y * mask_y).sum() / (mask_y.sum() + 1e-8)
        
        #==========================
        # 3. Total Loss = Masked L1 + λ * Smoothness
        #==========================
        total_loss = masked_loss_mean + self.lambda_smooth * smoothness_loss
        return total_loss