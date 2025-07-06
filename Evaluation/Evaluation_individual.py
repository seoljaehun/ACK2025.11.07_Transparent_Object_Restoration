# 불투명 VS 투명 물체의 MDE 성능 비교

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# .exr 이미지 파일을 openCV에 업로드하기위해 필요한 명령어
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로
gt_path = r"C:\VScode_Data\GT_image\000000060-output-depth.exr" # 정답 이미지(GT) 파일 경로
pred_A_path = r"C:\VScode_Data\Opaque_Depth_image\000000060-opaque-rgb-img-dpt_large_384.png" # 불투명 물체 뎁스 이미지 파일 경로
pred_B_path = r"C:\VScode_Data\Transparent_Depth_image\000000060-transparent-rgb-img-dpt_large_384.png" # 투명 물체 뎁스 이미지 파일 경로
mask_path = r"C:\VScode_Data\Mask_Depth_image\000000060-mask.png" # 투명 물체 Mask 이미지 파일 경로

# 총 4가지의 평가지표 정의
# RMSE: 루트 평균 오차 제곱 (낮을수록 좋음)
def rmse(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

# MAE: 평균 절대 오차 (낮을수록 좋음)
def mae(gt, pred):
    return np.mean(np.abs(gt - pred))

# abs REL: 절대 상대오차 (낮을수록 좋음)
def abs_rel(gt, pred):
    return np.mean(np.abs(gt - pred) / (gt + 1e-8))

# δ accuracy: 상대 오차 허용 기준 내 비율 (클수록 좋음)
def delta_accuracy(gt, pred, threshold=1.15): # 기준값 = 1.15
    ratio = np.maximum(gt / pred, pred / gt)
    return np.mean(ratio < threshold)

# 이미지 읽어오기
gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
pred_A = cv2.imread(pred_A_path, cv2.IMREAD_UNCHANGED)
pred_B = cv2.imread(pred_B_path, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 채널 분리
gt = gt[:, :, 0]
pred_A = pred_A[:, :, 0]
pred_B = pred_B[:, :, 0]

# float32로 변환
gt = gt.astype(np.float32)
pred_A = pred_A.astype(np.float32)
pred_B = pred_B.astype(np.float32)

# GT 이미지를 Mask 이미지와 같은 size로 변경
target_size = (mask.shape[1], mask.shape[0])
gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_LINEAR)

# Mask 이진 분류, 투명 물체 부분만 1값 나머지 0값
binary_mask = (mask == 255).astype(np.uint8)

# Mask 적용
gt_masked = gt[binary_mask == 1]
pred_A_masked = pred_A[binary_mask == 1]
pred_B_masked = pred_B[binary_mask == 1]

# Mask를 적용한 이미지
gt_masked_img = np.where(binary_mask == 1, gt, np.nan)
pred_A_masked_img = np.where(binary_mask == 1, pred_A, np.nan)
pred_B_masked_img = np.where(binary_mask == 1, pred_B, np.nan)

# 유효값 필터링
valid = (gt_masked > 0) & (pred_A_masked > 0) & (pred_B_masked > 0) & (~np.isnan(gt_masked)) & (~np.isnan(pred_A_masked)) & (~np.isnan(pred_B_masked))
g = gt_masked[valid]
p_A = pred_A_masked[valid]
p_B = pred_B_masked[valid]

# Scale + Shift 정규화
# 최소 제곱법으로 scale (s), shift (t) 계산,, 불투명 이미지 기준으로 계산
A = np.vstack([p_A, np.ones_like(p_A)]).T  # [N, 2] 행렬
s, t = np.linalg.lstsq(A, g, rcond=None)[0]

# 예측값 보정
p_A_aligned = s * p_A + t
p_B_aligned = s * p_B + t

# 평가지표 출력
print(f"Scale factor (s): {s:.6f}")
print(f"Shift (t):        {t:.6f}")
print("*** A Image Evaluation:")
print(f"RMSE       : {rmse(g, p_A_aligned):.4f}")
print(f"MAE        : {mae(g, p_A_aligned):.4f}")
print(f"AbsRel     : {abs_rel(g, p_A_aligned):.4f}")
print(f"δ < 1.15   : {delta_accuracy(g, p_A_aligned)*100:.4f}%")
print("*** B Image Evaluation:")
print(f"RMSE       : {rmse(g, p_B_aligned):.4f}")
print(f"MAE        : {mae(g, p_B_aligned):.4f}")
print(f"AbsRel     : {abs_rel(g, p_B_aligned):.4f}")
print(f"δ < 1.15   : {delta_accuracy(g, p_B_aligned)*100:.4f}%")

'''
plt.imshow(pred_masked_img, cmap='viridis')
plt.title('Defect Depth image')
plt.show()
'''