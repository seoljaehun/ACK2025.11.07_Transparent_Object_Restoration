#  Config 설정
# : 모든 경로와 하이퍼 파라미터를 중앙에서 관리하는 역할

import torch

class Config:
    #=====================
    # Dataset Path
    #=====================
    root_dir = r"D:\Dataset\Cleargrasp_Dataset"  # Cleargasp Dataset 상위 폴더 경로

    #=====================
    # Checkpoint Path
    #=====================
    checkpoint_dir = "./checkpoints" # 학습 중 모델의 가중치 저장 폴더 경로

    #=====================
    # Inference Result Path
    #=====================
    inference_output_dir = r"D:\Dataset\Cleargrasp_Dataset\inference_results"   # 추론 결과 이미지 저장 경로
    attention_weight_dir = r"D:\Dataset\Cleargrasp_Dataset\Attention_weight"    # attention weight 이미지 저장 경로
    
    #=====================
    # Training Settings
    #=====================
    epochs = 30              # 학습 반복 횟수
    batch_size = 8           # 배치 사이즈
    learning_rate = 5e-5     # optimizer 학습률
    num_workers = 8          # DataLoader에서 데이터 로드할 병렬 스레드 수

    #=====================
    # Model Settings
    #=====================
    in_channels = 4          # 입력 채널: RGB(3) + Init Depth(1)
    out_channels = 1         # 출력 채널: Residual Depth(1)

    #=====================
    # Device
    #=====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 -> cuda 설정