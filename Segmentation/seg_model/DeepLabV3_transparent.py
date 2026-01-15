# 1. 라이브러리 임포트
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# 2. 데이터셋 클래스 정의
class ClearGraspDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        self.mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("이미지와 마스크 수가 다릅니다.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 흑백 마스크 (0=배경, 255=투명물체)
        mask = (mask > 127).astype(np.uint8)  # 255 → 1 로 바꾸기

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# 3. 경로 설정
image_dir = "ClearGrasp_Dataset/cleargrasp-dataset-test-val/cleargrasp-dataset-test-val/synthetic-val/cup-with-waves-val/rgb-imgs"
mask_dir  = "ClearGrasp_Dataset/cleargrasp-dataset-test-val/cleargrasp-dataset-test-val/synthetic-val/cup-with-waves-val/segmentation-masks"

# 4. 데이터 변환 정의
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 5. 데이터셋 및 데이터로더 구성
dataset = ClearGraspDataset(image_dir, mask_dir, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)

# 6. 사전학습된 DeepLabV3+ 모델 로딩
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=2                       
).cuda()

# 7. 손실함수 및 최적화 함수
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 8. 학습 루프
def train_model(n_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            imgs, masks = imgs.cuda(), masks.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.cuda(), masks.cuda()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_cleargrasp_unet.pth")
            print("✅ Best model saved")

# 9. 시각화 함수
def visualize_predictions():
    model.eval()
    imgs, masks = next(iter(val_loader))
    imgs, masks = imgs.cuda(), masks.cuda()
    preds = model(imgs)
    preds = torch.argmax(preds, dim=1)

    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    for i in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(imgs[i])
        axs[0].set_title("Image")
        axs[1].imshow(masks[i], cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(preds[i], cmap='gray')
        axs[2].set_title("Prediction")
        for ax in axs: ax.axis('off')
        plt.show()

# 10. 실행
if __name__ == '__main__':
    train_model(n_epochs=10)
    visualize_predictions()
