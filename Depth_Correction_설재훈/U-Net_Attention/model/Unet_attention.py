#  Model 정의 함수
# : attention block이 포함된 U-Net 기반 네트워크

import torch
import torch.nn as nn
import torch.nn.functional as F

#==========================================
# Attention Blocks (Occlusion / Contact / Normal)
#==========================================

# 1. Occlusion edge attention block 정의
class OcclusionEdgeAttention(nn.Module):
    def __init__(self, in_channels, aux_channels=1):
        """
        Args:
            in_channels: U-Net feature의 채널 수
            aux_channels: Occlusion edge의 채널 수(= 1)
        """
        # nn.Module 초기화
        super(OcclusionEdgeAttention, self).__init__()
        
        # Occlusion edge 입력 인코딩 -> Occlusion edge 특징 맵 생성
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Attention weight 계산
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # 범위 [0, 1], 각 픽셀마다 가중치
        )

    def forward(self, x, aux):
        """
        Args:
            x: U-Net feature
            aux: Occlusion edge 이미지
        """
        aux_feat = self.aux_conv(aux)       # occlusion edge -> feaature 변환
        combined = x + aux_feat             # U-Net feature + Occlusion feature 결합
        weight = self.att_conv(combined)    # Attention weight 계산 ([0,1])
        self.att_weight = weight            # Attention weight Map 저장, (B, C, H, W)
        
        # 원래 feature에 가중치 적용
        return x * weight               

# 2. Contact edge attention block 정의
class ContactEdgeAttention(nn.Module):
    def __init__(self, in_channels, aux_channels=1):
        """
        Args:
            in_channels: U-Net feature의 채널 수
            aux_channels: Contact edge의 채널 수(= 1)
        """
        # nn.Module 초기화
        super(ContactEdgeAttention, self).__init__()
        
        # Contact edge 입력 인코딩 -> Contact edge 특징 맵 생성
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Attention weight 계산
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # 범위 [0, 1], 각 픽셀마다 가중치
        )

    def forward(self, x, aux):
        """
        Args:
            x: U-Net feature
            aux: Contact edge 이미지
        """
        aux_feat = self.aux_conv(aux)       # Contact edge -> feaature 변환
        combined = x + aux_feat             # U-Net feature + Contact feature 결합
        weight = self.att_conv(combined)    # Attention weight 계산 ([0,1])
        self.att_weight = weight            # Attention weight Map 저장, (B, C, H, W)
        
        # 원래 feature에 가중치 적용
        return x * weight

# 3. Normal attention block 정의
class NormalAttention(nn.Module):
    def __init__(self, in_channels, aux_channels=3):
        """
        Args:
            in_channels: U-Net feature의 채널 수
            aux_channels: Normal의 채널 수(= 3)
        """
        # nn.Module 초기화
        super(NormalAttention, self).__init__()
        
        # Normal 입력 인코딩 -> Normal 특징 맵 생성
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Attention weight 계산
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # 범위 [0, 1], 각 픽셀마다 가중치
        )

    def forward(self, x, aux):
        """
        Args:
            x: U-Net feature
            aux: Normal 이미지
        """
        aux_feat = self.aux_conv(aux)       # Normal edge -> feaature 변환
        combined = x + aux_feat             # U-Net feature + Normal feature 결합
        weight = self.att_conv(combined)    # Attention weight 계산 ([0,1])
        self.att_weight = weight            # Attention weight Map 저장, (B, C, H, W)
        
        # 원래 feature에 가중치 적용
        return x * weight


#==========================================
# Standard U-Net Building Blocks
#==========================================

# 1. DoubleConv block 정의
# : 공간 크기는 유지, 채널 수는 변화(증가, 감소 둘 다 가능)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 입력 피처 맵의 채널 수
            out_channels: 출력 피처 맵의 채널 수
        """
        super(DoubleConv, self).__init__()
        
        # Conv + BN + ReLU 연산 연속 2번 수행
        self.conv = nn.Sequential(
            # (1) 채널 변경: in_channels → out_channels, 특징 추출
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # (2) 채널 유지, 추가 특징 추출
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: 입력 feature map
        """
        return self.conv(x)

# 2. 다운 샘플링 block 정의
# : 공간 크기를 절반으로 줄이고, 채널 수를 2배 늘림
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 입력 피처 맵의 채널 수
            out_channels: 출력 피처 맵의 채널 수
        """
        super(Down, self).__init__()
        
        # 크기를 절반으로 줄이는 MaxPooling
        self.pool = nn.MaxPool2d(2)
        # DoubleConv -> 채널 수 2배 증가, 특징 추출
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: 입력 feature map
        """
        return self.conv(self.pool(x))

# 3. 업 샘플링 block 정의
# : 공간 크기를 2배 늘리고, 채널 수를 절반으로 줄임, 스킵 연결로 인코더의 정보를 합침
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: 입력 피처 맵의 채널 수
            skip_channels: 스킵 연결에서 가져올 인코더 피쳐 맵의 채널 수
            out_channels: 출력 피처 맵의 채널 수
        """
        super(Up, self).__init__()
        
        # 크기를 2배로 확장하고, 채널 수를 절반으로 줄임
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 입력: 절반으로 줄어든 채널 + 인코더 피쳐 채널
        # DoubleConv -> 다시 채널 수를 절반으로 감소, 특징 추출
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: 입력 feature map (디코더에서 내려온 feature)
            x2: 인코더에서 스킵 연결된 feature map
        """
        x1 = self.up(x1)    # 크기 2배 확장, 채널 수 절반 감소
        
        # 크기 맞춤 (패딩)
        # x1과 x2의 크기가 같이 않아, 두 feature의 크기를 동일하게 맞춤
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 채널 방향으로 합침
        # : 절반으로 줄어든 채널 + 인코더 피쳐 채널
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 4. OutConv block 정의
# : 공간 크기는 유지하고, 출력 채널을 원하는 개수로 변환
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 입력 피처 맵의 채널 수
            out_channels: 출력 피처 맵의 채널 수
        """
        super(OutConv, self).__init__()
        
        # 크기는 유지하고, 채널 압축/변환 (Depth Map: ? -> 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)  # Activation 없음 (Residual 학습)


#==========================================
# U-Net with Attention Integration
#==========================================
class UNetAttention(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        """
        Args:
            n_channels: 입력 피처 맵의 채널 수 (RGB + init_Depth = 4)
            n_classes: 출력 피처 맵의 채널 수 (Depth = 1)
        """
        # Encoder 인스턴트 변수 정의
        super(UNetAttention, self).__init__() 
        self.inc = DoubleConv(n_channels, 64) # (4, 288, 512) -> (64, 288, 512)
        self.down1 = Down(64, 128)      # down1: (64, 288, 512) -> (128, 144, 256)
        self.down2 = Down(128, 256)     # down2: (128, 144, 256) -> (256, 72, 128)
        self.down3 = Down(256, 512)     # down3: (256, 72, 128) -> (512, 36, 64)
        self.down4 = Down(512, 512)     # down4: (512, 36, 64) -> (512, 18, 32)

        # Decoder 인스턴트 변수 정의
        self.up1 = Up(512, 512, 256)    # up1: (512, 18, 32) -> (256, 36, 64)
        self.up2 = Up(256, 256, 128)    # up2: (256, 36, 64) -> (128, 72, 128)
        self.up3 = Up(128, 128, 64)     # up3: (128, 72, 128) -> (64, 144, 256)
        self.up4 = Up(64, 64, 64)       # up4: (64, 144, 256) -> (64, 288, 512)
        self.outc = OutConv(64, n_classes) # (64, 288, 512) -> (1, 288, 512)

        # Attention Blocks 생성
        self.occ_edge_att = OcclusionEdgeAttention(64, aux_channels=1)
        self.contact_edge_att = ContactEdgeAttention(256, aux_channels=1)
        self.normal_att = NormalAttention(512, aux_channels=3)

    def forward(self, x, occ_edge=None, contact_edge=None, normal_img=None):
        """
        Args:
            x: 메인 입력 (RGB + init_Depth = 4채널)
            occ_edge: Occlusion edge 이미지 (1채널)
            contact_edge: Contact edge 이미지 (1채널)
            normal_img: Normal 이미지 (3채널)
        Returns:
            logits: residual depth map, (B, 1, H, W)
        """
        # Encoder
        # 공간 크기가 크면 해상도가 높아져 세부 구조 특징 훈련
        # 공간 크기가 줄어들면 해상도가 낮아져 추상적인 특징 훈련
        
        # 1. Occlusion edge: 경계 부분의 픽셀 위치가 중요 -> 초기 레이어에서 고해상도로 반영
        # 2. Contact edge: 물체와 배경의 접촉점에서 상호작용이 중요 -> 중간 레이어에서 중간해상도로 반영
        # 3. Normal: 법선 백터는 전역 구조를 이해하는데 중요 -> 깊은 레이어에서 저해상도로 반영
        
        
        x1 = self.inc(x)        # (Batch, 4, 288, 512) -> (Batch, 64, 288, 512)
        
        # Occlusion edge attention block 적용
        if occ_edge is not None:
            # occ_edge의 공간 크기를 x1에 맞춤
            occ_edge = F.interpolate(occ_edge, size=x1.shape[2:], mode='bilinear', align_corners=False)
            # x1에 attention 적용하여 가중치를 곱함
            x1 = self.occ_edge_att(x1, occ_edge)

        x2 = self.down1(x1)     # (Batch, 64, 288, 512) -> (Batch, 128, 144, 256)
        x3 = self.down2(x2)     # (Batch, 128, 144, 256) -> (Batch, 256, 72, 128)
        
        # Contact edge attention block 적용
        if contact_edge is not None:
            # contact_edge의 공간 크기를 x3에 맞춤
            contact_edge = F.interpolate(contact_edge, size=x3.shape[2:], mode='bilinear', align_corners=False)
            # x3에 attention 적용하여 가중치를 곱함
            x3 = self.contact_edge_att(x3, contact_edge)

        x4 = self.down3(x3)     # (Batch, 256, 72, 128) -> (Batch, 512, 36, 64)
        x5 = self.down4(x4)     # (Batch, 512, 36, 64) -> (Batch, 512, 18, 32)
        
        # Normal attention block 적용
        if normal_img is not None:
            # normal_img의 공간 크기를 x5에 맞춤
            normal_img = F.interpolate(normal_img, size=x5.shape[2:], mode='bilinear', align_corners=False)
            # x5에 attention 적용하여 가중치를 곱함
            x5 = self.normal_att(x5, normal_img)

        # Decoder
        x = self.up1(x5, x4)    # (Batch, 512, 18, 32) -> (Batch, 256, 36, 64)
        x = self.up2(x, x3)     # (Batch, 256, 36, 64) -> (Batch, 128, 72, 128)
        x = self.up3(x, x2)     # (Batch, 128, 72, 128) -> (Batch, 64, 144, 256)
        x = self.up4(x, x1)     # (Batch, 64, 144, 256) -> (Batch, 64, 288, 512)
        logits = self.outc(x)   # (Batch, 64, 288, 512) -> (Batch, 1, 288, 512)
        return logits
    