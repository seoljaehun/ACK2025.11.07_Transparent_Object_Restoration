#  Model 정의 함수
# : Surface Normal을 추출하기 위한 DeepLabV3+ 기반 네트워크

import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DeepLabv3Plus_for_Normals(nn.Module):
    """
        - 백본(Backbone): 사전 학습된 ResNet-50 사용
        - input channels: 3 (RGB 이미지)
        - output channels: 3 (Surface Normal 벡터)
    """
    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): ResNet-50 백본에 ImageNet 사전 학습 가중치를 사용할지 여부
        """
        super(DeepLabv3Plus_for_Normals, self).__init__()

        # 1. torchvision에서 사전 학습된 DeepLabv3+ 모델 로드 (백본: ResNet-50)
        # weights 파라미터를 사용하여 최신 버전의 가중치 불러오기
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)

        # 2. 최종 출력 레이어(Classifier)를 새로운 Conv2d 레이어로 교체
        # 기존 모델의 출력 채널은 21 (PASCAL VOC 클래스 수)
        # 이를 서피스 노멀(Nx, Ny, Nz)을 위한 3채널로 변경
        self.deeplab.classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """
            모델의 순전파 과정을 정의
        Args:
            x (torch.Tensor): 입력 이미지 텐서 (Batch, 3, Height, Width)

        Returns:
            torch.Tensor: 정규화된 서피스 노멀 맵 (Batch, 3, Height, Width)
        """
        # 3. 입력 이미지를 모델에 통과시켜 예측 결과를 얻음
        # torchvision 모델은 출력을 딕셔너리 형태('out', 'aux')로 반환
        raw_output = self.deeplab(x)['out']

        # 4. L2 정규화를 통해 각 픽셀의 벡터를 단위 벡터(크기=1)로 변환
        normalized_output = F.normalize(raw_output, p=2, dim=1)

        return normalized_output
