#  L1 Loss 함수
# : 딥러닝 모델의 Loss 함수 정의, L1 Loss + Gradient Consistency Loss + Surface Normal Consistency Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.scale_alignment import align_scale  # utils -> scale_alignment -> align_scale 함수 불러오기

# Loss 정의
class L1Loss(nn.Module):
    """
        L1 Loss + Gradient Consistency Loss + Surface Normal Consistency Loss
        1. L1 Loss: 예측된 Depth Map의 예측 정확도를 높이기 위한 Loss
        2. Gradient Consistency Loss: 예측된 Depth Map과 GT Depth Map의 기울기 구조를 비슷하게 만들기 위한 Loss
        3. Surface Normal Consistency Loss: 예측된 Depth Map과 GT Depth Map의 표면 법선 벡터의 방향을 비슷하게 만들기 위한 Loss
    """
    def __init__(self, lambda_Gradient=1.0, lambda_Normal=0.01): 
        # 부모 클래스(nn.Module) 초기화
        super(L1Loss, self).__init__()
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
    
    def compute_normal_from_depth(self, depth_map, scaling_factor):
        """
            깊이 맵과 스케일링 팩터를 받아 표면 법선을 계산합니다.
            depth_map: (B, 1, H, W) torch tensor
            return: estimated_normal: (B, 3, H, W)
        """
        device = depth_map.device
        scaled_depth = depth_map * scaling_factor

        # Sobel 필터 커널 정의
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
        # 컨볼루션을 이용해 그래디언트(깊이 변화량) 계산
        dz_dx = F.conv2d(scaled_depth, sobel_x, padding='same')
        dz_dy = F.conv2d(scaled_depth, sobel_y, padding='same')

        # 법선 벡터 구성: n = [-dz/dx, -dz/dy, 1]
        n_x = -dz_dx
        n_y = -dz_dy
        # z 성분 항상 1.0으로 고정
        n_z = torch.ones_like(n_x)

         # (n_x, n_y, n_z)를 하나의 3D normal 벡터로 합치기
        normal = torch.cat([n_x, n_y, n_z], dim=1)
    
        # 최종적으로 벡터를 정규화하여 크기를 1로 만듦(단위 벡터)
        estimated_normal = F.normalize(normal, dim=1, eps=1e-6)
    
        return estimated_normal

    def forward(self, pred, target):
        """
            총 손실 값 계산
        Args:
            pred: 모델 뎁스 예측값, (B, 1, H, W)
            target: gt 뎁스 값, (B, 1, H, W)
        """

        # 예측 depth 이미지를 gt_depth 이미지에 대한 s, t 값 계산
        s, t = align_scale(target, pred)
        
        # s와 t의 모양을 (B,) -> (B, 1, 1, 1)로 바꿔 pred와 연산할 수 있게 함
        s_view = s.view(-1, 1, 1, 1)
        t_view = t.view(-1, 1, 1, 1)
        
        aligned_pred = s_view * pred + t_view
        
        #==========================
        # 1. basic L1 Loss
        #==========================
        
        # 예측값과 gt 값의 절대 차이 계산
        loss_L1 = torch.abs(aligned_pred - target)
        
        # 모든 픽셀의 Loss 평균 값 
        loss_L1_mean = loss_L1.mean()
        
        #==========================
        # 2. Gradient Consistency Loss
        #==========================
        
        # 예측 Depth Map과 GT Depth Map의 Gradient 계산
        pred_dx = self.gradient_x(aligned_pred)
        pred_dy = self.gradient_y(aligned_pred)
        target_dx = self.gradient_x(target)
        target_dy = self.gradient_y(target)
        
        # Gradient 차이 계산
        Grad_loss_x = torch.abs(pred_dx - target_dx)
        Grad_loss_y = torch.abs(pred_dy - target_dy)
        
        # Gradient Loss Map 계산
        Gradient_loss_mean = (Grad_loss_x.mean() + Grad_loss_y.mean()) / 2
        
        #==========================
        # 3. Surface Normal Consistency Loss
        #==========================
        
        SCALING_FACTOR = 500
        
        # Initial Depth로부터 Predicted Normal 계산
        pred_normal = self.compute_normal_from_depth(aligned_pred, scaling_factor=SCALING_FACTOR)
        # GT Depth로부터 Target Normal 계산
        target_normal = self.compute_normal_from_depth(target, scaling_factor=SCALING_FACTOR)
        
        # cosine similarity 계산
        # : 두 벡터의 내적 -> pred_normal ⋅ gt_normal = cos(θ)
        cos_sim = torch.sum(pred_normal * target_normal, dim=1, keepdim=True)  # (B, 1, H, W)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 안정성 확보를 위해 -1 ~ 1 사이로 clamp
        
        # Loss = cosine 유사도 거리: 1 - cos(θ) -> 0 이면 같은 방향, 1 이면 완전 다른 방향
        normal_loss = 1.0 - cos_sim     # (B, 1, H, W)
        
        # Normal Loss Map 계산 (배경 영역의 Loss 값은 0으로 무시)
        normal_loss_mean = normal_loss.mean()
        
        #==========================
        # 4. Total Loss = Masked L1 + λ1 * Gradient Loss + λ2 * Normal Loss
        #==========================
        total_loss = loss_L1_mean + self.lambda_Gradient * Gradient_loss_mean + self.lambda_Normal * normal_loss_mean
        return total_loss, loss_L1_mean, Gradient_loss_mean, normal_loss_mean