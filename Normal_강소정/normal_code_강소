#준비 및 경로 설정 
# !nvidia-smi  # (선택) GPU 확인
!pip -q install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
from pathlib import Path
import os, zipfile, glob, math, random, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# === 사용자 설정 ===
# 구글 드라이브에 올려둔 zip 경로 또는 Colab에 업로드한 zip 경로
RGB_ZIPS = [
    "/content/rgb.zip",                 # 예: "/content/drive/MyDrive/datasets/rgb.zip"
]
NORM_ZIPS = [
    "/content/camera-normals.zip",      # 예: "/content/drive/MyDrive/datasets/normals.zip"
]

OUT_DIR = Path("/content/data")
RGB_DIR = OUT_DIR/"rgb"
NORM_DIR = OUT_DIR/"normals"
CKPT_DIR = Path("/content/checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#노멀 디코드 , 에지맵
# 노멀맵 디코딩: RGB[0..255] -> N[-1..1], 유닛노름 보정
def decode_normal(img_np_uint8):
    n = img_np_uint8.astype(np.float32) / 255.0
    n = n*2.0 - 1.0   # [-1,1]
    # 유닛노름 보정
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    n = n / norm
    return n

# Sobel 에지 (간단한 스칼라 에지 강도) -> [H,W] float32 0..1
# 노멀맵 디코딩: RGB[0..255] -> N[-1..1], 유닛노름 보정
def decode_normal(img_np_uint8):
    n = img_np_uint8.astype(np.float32) / 255.0
    n = n*2.0 - 1.0   # [-1,1]
    # 유닛노름 보정
    norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    n = n / norm
    return n

# Sobel 에지 (간단한 스칼라 에지 강도) -> [H,W] float32 0..1
def sobel_edge(gray_t):
    # gray_t: (1,H,W) 0..1
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=gray_t.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=gray_t.device).view(1,1,3,3)
    gx = F.conv2d(gray_t, kx, padding=1)
    gy = F.conv2d(gray_t, ky, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + 1e-6)
    g  = g / (g.amax(dim=[2,3], keepdim=True)+1e-6)
    return g  # (1,H,W) 0..1

# Dataset & Dataloader
class NormalDataset(Dataset):
    def __init__(self, keys, rgb_map, norm_map, img_size=512, augment=True):
        self.keys = keys
        self.rgb_map = rgb_map
        self.norm_map = norm_map
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.color_jitter = transforms.ColorJitter(0.2,0.2,0.2,0.05)

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        rgb_p  = self.rgb_map[k]
        norm_p = self.norm_map[k]

        rgb = Image.open(rgb_p).convert("RGB")
        gt  = Image.open(norm_p).convert("RGB")

        if self.augment:
            if random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                gt  = gt.transpose(Image.FLIP_LEFT_RIGHT)

        if self.augment and random.random() < 0.8:
            rgb = self.color_jitter(rgb)

        rgb_t = self.to_tensor(rgb)              # [3,H,W] 0..1
        rgb_t = self.resize(rgb_t)
        gray  = rgb_t.mean(0, keepdim=True)      # [1,H,W]

        gt_np = np.array(self.resize(self.to_tensor(gt)).permute(1,2,0).numpy()*255, dtype=np.uint8)
        n_np  = decode_normal(gt_np)             # [H,W,3] float32 [-1,1]
        n_t   = torch.from_numpy(n_np).permute(2,0,1)  # [3,H,W]

        return {
            "image": rgb_t,
            "gray": gray,
            "normal": n_t
        }

# split
random.shuffle(common)
split = int(len(common)*0.9)
train_keys, val_keys = common[:split], common[split:]

train_ds = NormalDataset(train_keys, rgb_idx, norm_idx, img_size=512, augment=True)
val_ds   = NormalDataset(val_keys,   rgb_idx, norm_idx, img_size=512, augment=False)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

len(train_ds), len(val_ds)

# 모델(DeepLabv3 + Edge Attention)
class EdgeAttentionBlock(nn.Module):
    """
    간단한 Spatial Attention:
    - 입력: DeepLab encoder feature(Ch), edge map(1)
    - 출력: 같은 크기 feature * attention
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+1, in_ch//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, feat, edge1x):
        # edge를 feat 크기에 맞게 다운/업샘플
        e = F.interpolate(edge1x, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.conv(torch.cat([feat, e], dim=1))
        return feat * (1 + attn)  # edge 주변을 >1배 가중

class DeeplabNormal(nn.Module):
    def __init__(self, backbone='resnet50', out_ch=3):
        super().__init__()
        if backbone == 'resnet50':
            self.net = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
        else:
            self.net = models.segmentation.deeplabv3_resnet101(weights=None, weights_backbone=None)
        # classifier 채널 수 확인
        enc_ch = 256  # ASPP 출력 채널(DeepLabv3 기본)
        self.edge_attn = EdgeAttentionBlock(enc_ch)
        # 3채널 회귀
        self.net.classifier = nn.Sequential(
            *list(self.net.classifier.children())[:-1],  # ASPP까지 유지
            nn.Conv2d(enc_ch, enc_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, out_ch, 1)
        )
    def forward(self, x, edge1x):
        feats = self.net.backbone(x)['out']           # encoder feat
        feats = self.edge_attn(feats, edge1x)         # edge attention
        out   = self.net.classifier(feats)            # logits (3)
        out   = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # 단위 노멀로 정규화
        out   = F.normalize(out, dim=1)
        return out

model = DeeplabNormal(backbone='resnet50').to(device)
sum(p.numel() for p in model.parameters())/1e6

#손실/지표
def cosine_loss(pred, target, mask=None):
    # pred,target: [B,3,H,W], unit-normalized
    cos = torch.clamp((pred*target).sum(1, keepdim=True), -1+1e-6, 1-1e-6)
    loss = 1 - cos
    if mask is not None:
        loss = loss*mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum()/denom
    return loss.mean()

def l1_loss(pred, target, mask=None):
    l = (pred - target).abs()
    if mask is not None:
        l = l*mask
        denom = mask.sum().clamp(min=1.0)
        return l.sum()/denom
    return l.mean()

@torch.no_grad()
def mean_angular_error_deg(pred, target):
    # expects unit vectors
    cos = torch.clamp((pred*target).sum(1), -1+1e-6, 1-1e-6)
    ang = torch.acos(cos) * 180.0 / math.pi
    return ang.mean().item()

#학습루프 
epochs = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
best_val = 1e9

def make_edge_from_gray(gray):
    # gray: [B,1,H,W] 0..1
    e_list = []
    for i in range(gray.shape[0]):
        e = sobel_edge(gray[i:i+1].to(device))    # (1,1,H,W)
        e_list.append(e)
    return torch.cat(e_list, dim=0)               # (B,1,H,W)

for epoch in range(1, epochs+1):
    model.train()
    t0 = time.time()
    total = 0.0
    for batch in train_dl:
        img  = batch["image"].to(device, non_blocking=True)
        gray = batch["gray"].to(device, non_blocking=True)
        gt   = batch["normal"].to(device, non_blocking=True)

        edge = make_edge_from_gray(gray)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            pred = model(img, edge)
            loss = cosine_loss(pred, gt) + 0.2*l1_loss(pred, gt)
            # 유닛노름 정규화 페널티(미세)
            unit_pen = (pred.norm(dim=1) - 1.0).abs().mean()
            loss = loss + 0.01*unit_pen

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0; val_ang = 0.0; n=0
    with torch.no_grad():
        for batch in val_dl:
            img  = batch["image"].to(device, non_blocking=True)
            gray = batch["gray"].to(device, non_blocking=True)
            gt   = batch["normal"].to(device, non_blocking=True)
            edge = make_edge_from_gray(gray)
            pred = model(img, edge)
            val_loss += (cosine_loss(pred, gt) + 0.2*l1_loss(pred, gt)).item()
            val_ang  += mean_angular_error_deg(pred, gt)
            n += 1
    val_loss /= max(n,1); val_ang/=max(n,1)

    print(f"[{epoch:02d}/{epochs}] train_loss={total/len(train_dl):.4f}  "
          f"val_loss={val_loss:.4f}  MAE(deg)={val_ang:.2f}  ({time.time()-t0:.1f}s)")

    # 체크포인트
    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = CKPT_DIR/f"deeplab_normals_best.pt"
        torch.save({"epoch":epoch,
                    "state_dict":model.state_dict(),
                    "val_loss":val_loss,
                    "val_mae":val_ang}, ckpt_path)
        print("  -> saved:", ckpt_path)

#추론/시각화(테스트 1장)
import matplotlib.pyplot as plt

@torch.no_grad()
def infer_one(pil_img):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512), antialias=True),
    ])(pil_img).unsqueeze(0).to(device)
    gray = t.mean(1, keepdim=True)
    edge = make_edge_from_gray(gray)
    pred = model(t, edge)[0].cpu().permute(1,2,0).numpy()
    # [-1,1] -> 시각화용 0..1
    vis = (pred*0.5 + 0.5).clip(0,1)
    return vis

# 예시: 검증셋 첫 이미지
sample_key = val_keys[0]
img = Image.open(rgb_idx[sample_key]).convert("RGB")
vis = infer_one(img)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.title("RGB"); plt.imshow(img); plt.axis('off')
plt.subplot(1,2,2); plt.title("Pred Normal (viz)"); plt.imshow(vis); plt.axis('off')
plt.show()



