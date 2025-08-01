#  Mask-L1 Loss 함수
# : 딥러닝 모델의 Loss 함수 정의, Masked Smooth L1 Loss + Gradient Consistency Loss + Surface Normal Consistency Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss 정의
class MaskedL1Loss(nn.Module):
    """
        Masked Smooth L1 Loss + Gradient Consistency Loss + Surface Normal Consistency Loss
        1. Masked Smooth L1 Loss: 예측된 Depth Map의 오차 예측 정확도를 높이기 위한 Loss
        2. Gradient Consistency Loss: 예측된 Depth Map과 GT Depth Map의 기울기 구조를 비슷하게 만들기 위한 Loss
        3. Surface Normal Consistency Loss: 예측된 Depth Map과 Normal Map의 표면 법선 벡터의 방향을 비슷하게 만들기 위한 Loss
    """
    def __init__(self, beta=1.0, lambda_Gradient=0.05, lambda_Normal=0.05): 
        # 부모 클래스(nn.Module) 초기화
        super(MaskedL1Loss, self).__init__()
        self.beta = beta    # beta: Smooth L1의 Huber Loss 전환점
        self.lambda_Gradient = lambda_Gradient  # Gradient Consistency Loss의 가중치
        self.lambda_Normal = lambda_Normal      # Surface Normal Consistency Loss의 가중치

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
    
    def forward(self, pred, target, init, mask, normal):
        """
            총 손실 값 계산
        Args:
            pred: 모델 오차 예측값, (B, 1, H, W)
            target: gt 오차값(residual), (B, 1, H, W)
            init: 초기 뎁스 값, (B, 1, H, W)
            mask: 투명 영역 segmentation mask (1 = valid, 0 = invalid), (B, 1, H, W)
            normal: 표면 법선 벡터 값 -> GT normal 역할 (B, 3, H, W)
        """
        # 예측 뎁스, GT 뎁스 계산
        pred_Depth = init + pred
        target_Depth = init + target
        
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
        # 2. Gradient Consistency Loss
        #==========================
        
        # 예측 Depth Map과 GT Depth Map의 Gradient 계산
        pred_dx = self.gradient_x(pred_Depth)
        pred_dy = self.gradient_y(pred_Depth)
        target_dx = self.gradient_x(target_Depth)
        target_dy = self.gradient_y(target_Depth)
        
        # Gradient 차이 계산
        Grad_loss_x = torch.abs(pred_dx - target_dx)
        Grad_loss_y = torch.abs(pred_dy - target_dy)
        
        # mask는 gradient 크기에 맞게 슬라이스 (W-1, H-1 차이 보정)
        mask_x = mask[:, :, :, :-1]
        mask_y = mask[:, :, :-1, :]
        
        # Gradient Loss Map 계산 (배경 영역의 Loss 값은 0으로 무시)
        Gradient_loss_mean = ((Grad_loss_x * mask_x).sum() / (mask_x.sum() + 1e-6) + (Grad_loss_y * mask_y).sum() / (mask_y.sum() + 1e-6)) / 2
        
        #==========================
        # 3. Surface Normal Consistency Loss
        #==========================
        
        # pred Gradient를 기반으로 예측 normal 성분 계산, 크기 보정을 위해 padding 적용
        # n = [-dz/dx, -dz/dy, 1] 을 정규화
        n_x = F.pad(-pred_dx, (0,1,0,0), mode='replicate')  # x 성분
        n_y = F.pad(-pred_dy, (0,0,0,1), mode='replicate')  # y 성분
        n_z = torch.ones_like(n_x)      # z 성분은 항상 1로 설정
        
        # (n_x, n_y, n_z)를 하나의 3D normal 벡터로 합치기
        pred_normal = torch.cat([n_x, n_y, n_z], dim=1)  # (B, 3, H, W)
        
        # 각 위치의 벡터를 단위 벡터로 정규화 (norm = 1)
        pred_normal = F.normalize(pred_normal, dim=1, eps=1e-6)
        # GT normal 벡터를 단위 벡터로 정규화 (norm = 1)
        gt_normal = F.normalize(normal, dim=1, eps=1e-6)
        
        # cosine similarity 계산
        # : 두 벡터의 내적 -> pred_normal ⋅ gt_normal = cos(θ)
        cos_sim = torch.sum(pred_normal * gt_normal, dim=1, keepdim=True)  # (B, 1, H, W)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 안정성 확보를 위해 -1 ~ 1 사이로 clamp
        
        # Loss = cosine 유사도 거리: 1 - cos(θ) -> 0 이면 같은 방향, 1 이면 완전 다른 방향
        normal_loss = 1.0 - cos_sim     # (B, 1, H, W)
        
        # Normal Loss Map 계산 (배경 영역의 Loss 값은 0으로 무시)
        normal_loss_mean = (normal_loss * mask).sum() / (mask.sum() + 1e-6)
        
        #==========================
        # 4. Total Loss = Masked L1 + λ1 * Gradient Loss + λ2 * Normal Loss
        #==========================
        total_loss = masked_loss_mean + self.lambda_Gradient * Gradient_loss_mean + self.lambda_Normal * normal_loss_mean
        return total_loss