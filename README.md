# 🥇 ACK2025_Transparent_Object_Restoration

---
### 휴머노이드 로봇을 위한 투명 물체 깊이 복원 알고리즘

깊이 인식은 휴머노이드 로봇의 파지 및 자율주행을 위한 핵심 기술로, 최근에는 RGB 이미지 기반의 단안 깊이 추정(MDE) 기술이 활발히 연구되고 있다.
하지만 빛의 굴절, 투과 특성을 가진 투명 물체에는 정확도가 현저히 저하되는 문제가 있다. 
이에 본 논문에서는 MDE로 추정한 초기 깊이를 기반으로, 투명 물체의 깊이를 복원하는 U-NET 기반 알고리즘을 제안한다.
제안하는 알고리즘은 초기 깊이 외에도 보조 시각 정보를 Attention 구조 및 손실 함수에 적용하여 객체의 경계 및 형태를 정밀하게 복원한다.
실험 결과, 기존 MDE 모델 대비 투명 물체 영역에서 MAE, RMSE 등 평가지표가 평균 3.98배 개선되었으며, 투명 물체 깊이 복원의 실용 가능성을 효과적으로 입증하였다.

---

# 1. 데이터 셋
- Google의 "**ClearGrasp Dataset**" 이미지 데이터

  - **Train/Validation/Test**
    
    > RGB Image : 45454/705/521장 (.PNG)
    >
    > Ground-Truth Depth :  45454/705/521장 (.EXR)
    >
    > Edge Map : 45454/705/521장 (.PNG)
    >
    > Surface Normal : 45454/705/521장 (.EXR)
    >
    > Segmentation Mask : 45454/705/521장 (.PNG)

# 2. 문제 제기

대표적인 MDE 모델(MiDaS DPT-large 384)을 활용하여 투명 물체 영역에서의 성능 한계를 실험적으로 분석

그림

- 불투명 물체 경우, 실제 형태와 구조가 잘 반영된 깊이 값으로 예측
- 투명 물체 경우, 물체 내부 또는 경계에서 깊이가 결손되거나 배경의 깊이 값으로 잘못 예측

->  기존 MDE 모델이 투명 물체의 광학적 특성을 반영하지 못하여, 해당 영역에서의 추정 정확도가 크게 저하됨

# 3. 알고리즘 구조

그림

+ **MDE Depth Estimation**

  - 기존 MDE 모델을 활용해 RGB 이미지로부터 Init Depth 생성

+ **Obtaining Auxiliary Visual Cues**
  
  - 동일한 RGB 이미지로부터 보조 시각 정보 추출
  - 보조 시각 정보: Occlusion Edge, Contact Edge, Surface Normal, Segmentation Mask
 
+ **Depth Restoration**
  
  - 복원 모델은 RGB 이미지와 초기 깊이 맵을 입력 받아 투명 영역의 결손이 복원된 최종 깊이 맵 생성
    
    > Occlusion/Contact Edge, Surface Normal : Attention 구조로 통합되어 특징 가중치 부여
    >
    > Segmentation Mask: 손실 함수 계산 시 마스킹 정보로 활용

# 4. 알고리즘 구현

**1. Depth Estimation**

MDE 모델(MiDaS DPT-large 384)을 활용해 Init Depth 생성

그림

**2. Auxiliary Visual Cues**

각각 독립된 딥러닝 서브 모델을 활용해 보조 시각 정보 추출

- Occlusion/Contact Edge : U-Net 모델 활용, Cross-Entropy 손실 함수를 적용하여 클래스 불균형 문제 완화
- Surface Normal : DeepLabV3+ 모델 활용, ImageNet으로 사전 학습된 ResNet-50을 백본으로, 전이 학습 수행
- Segmentation Mask : U-Net++ 모델 활용

그림

Occlusion Edge 88%, Contact Edge 72%의 분류 정확도, Segmentation Mask 98.21%의 픽셀 정확도, Sueface Normal 4.89°의 평균 각도 오차 달성

-> 깊이 복원에 활용될 수 있을 만큼 높은 정확도로 추출됨

**2. Depth Restoration Model**

투명 물체의 깊이 오차(Residual)을 추정하고, 이를 Init Depth와 합산하여 복원된 깊이 맵 생성

그림

- U-Net 기반의 Encoder-Decoder 구조
- RGB Image + Init Depth = 4채널 입력, Residual Depth Map = 1채널 출력
- 보조 시각 정보를 각기 다른 Encoder 단계에 통합 및 손실 함수 적용
  
  + Occlusion Edge : 64채널에 적용하여, 물체-배경 깊이 불연속성 강조
  + Contact Edge : 256채널에 적용하여, 물체-지면 깊이 연속성 확보
  + Surface Normal : 512채널에 적용하여, 곡률과 구조의 기하학적 일관성 유지
  + Segmentation Mask : 손실 함수 계산에 마스킹 정보로 곱해져, 배경 영역의 불필요한 오차 반역을 줄이고, 복원 효율 강화

# 4. 실험 결과

복원된 Depth를 Init Depth와 비교하고, Ablation Study를 통해 핵심 구성 요소의 효과 분석

그림

- 제안 모델이 모든 평가지표에서 가장 우수한 성능을 보임

제안 모델이 기존 MDE 모델 대비 투명 영역의 결손을 현저히 감소시켰고, Ablation Study를 통해 Attention 모듈과 오차 학습 방식의 효과를 입증하였음.

---

# 관련 자료

- Paper : <>
- Dataset : <https://sites.google.com/view/cleargrasp/data?authuser=0>
- 참고 문헌 : <>
