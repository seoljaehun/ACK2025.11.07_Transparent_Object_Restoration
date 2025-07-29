# 불투명 VS 투명 물체의 MDE 성능 비교(individual)
# 투명 물체 부분만 정규화

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# .exr 이미지 파일을 openCV에 업로드하기 위해 필요한 명령어
import cv2
import numpy as np
import matplotlib.pyplot as plt

""" PFM 파일을 읽어 numpy float32 배열 반환. """
def load_pfm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':  # color image (3 channels)
            color = True
        elif header == 'Pf':  # grayscale image (1 channel)
            color = False
        else:
            raise ValueError('Not a PFM file.')

        # Read width and height
        dim_line = ''
        while True:
            line = f.readline().decode('utf-8')
            if line.startswith('#'):
                continue  # skip comments
            else:
                dim_line = line
                break
        width, height = map(int, dim_line.strip().split())

        # Read scale factor (and endianness)
        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'  # little or big endian
        scale = abs(scale)

        # Read pixel data
        num_channels = 3 if color else 1
        data = np.fromfile(f, endian + 'f')  # float32
        shape = (height, width, num_channels) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored bottom to top

        return data, scale
    
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
def delta_accuracy(gt, pred, threshold=1.05): # 기준값 = 1.05
    ratio = np.maximum(gt / pred, pred / gt)
    return np.mean(ratio < threshold)

# 파일 경로
gt_path = r'C:\SJH\Python\Transparent_Object_Detection\Data\Opaque_Transparent_Comparison\GT_image\000000000-output-depth.exr' # 정답 이미지(GT) 파일 경로
opaque_path = r'C:\SJH\Python\Transparent_Object_Detection\Data\Opaque_Transparent_Comparison\Opaque_image\000000000-opaque-rgb-img-dpt_large_384.pfm' # 불투명 물체 뎁스 이미지 파일 경로
transp_path = r'C:\SJH\Python\Transparent_Object_Detection\Data\Opaque_Transparent_Comparison\Transparent_image\000000000-transparent-rgb-img-dpt_large_384.pfm' # 투명 물체 뎁스 이미지 파일 경로
mask_path = r'C:\SJH\Python\Transparent_Object_Detection\Data\Opaque_Transparent_Comparison\Mask_image\000000000-mask.png' # 투명 물체 Mask 이미지 파일 경로

# 이미지 읽어오기
gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
opaque, scale_A = load_pfm(opaque_path)
transp, scale_B = load_pfm(transp_path)

# GT 이미지를 Mask 이미지와 같은 size로 변경(up)
target_size = (mask.shape[1], mask.shape[0])
gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_LINEAR)

# Mask 이진 분류, 투명 물체 부분만 1값 나머지 0값
binary_mask = (mask == 255).astype(np.uint8)

# 투명 물체 Mask 적용 
opaque_object = opaque[binary_mask == 1]
gt_object = gt[binary_mask == 1]

# valid 픽셀 마스크 (gt>0, pred>0)
valid = (gt_object > 0) & (opaque_object > 0)
p = opaque_object[valid]        # 예측 상대적인 disparity
g = gt_object[valid]            # GT

# Scale + Shift 정규화
# 최소 제곱법으로 scale (s), shift (t) 계산,, 불투명 이미지 기준
A = np.vstack([p, np.ones_like(p)]).T  # [N, 2] 행렬
s, t = np.linalg.lstsq(A, g, rcond=None)[0]
print(f"정렬 계수: s={s:.4f}, t={t:.4f}")

# 예측값 보정
opaque_aligned = np.clip(s * opaque + t, 1e-8, None)
transp_aligned = np.clip(s * transp + t, 1e-8, None)

# 투명 물체 Mask 적용
gt_masked = gt[binary_mask == 1]
opaque_masked = opaque_aligned[binary_mask == 1]
transp_masked = transp_aligned[binary_mask == 1]

# 마스킹 영역에서 유효값 필터링
valid = (gt_masked > 0) & (opaque_masked > 0) & (transp_masked > 0)
g = gt_masked[valid]
p_opaque = opaque_masked[valid]
p_transp = transp_masked[valid]

# 평가지표 출력
print("*** A Image Evaluation:")
print(f"RMSE       : {rmse(g, p_opaque):.4f}")
print(f"MAE        : {mae(g, p_opaque):.4f}")
print(f"AbsRel     : {abs_rel(g, p_opaque):.4f}")
print(f"δ < 1.05   : {delta_accuracy(g, p_opaque)*100:.4f}%")
print("*** B Image Evaluation:")
print(f"RMSE       : {rmse(g, p_transp):.4f}")
print(f"MAE        : {mae(g, p_transp):.4f}")
print(f"AbsRel     : {abs_rel(g, p_transp):.4f}")
print(f"δ < 1.05   : {delta_accuracy(g, p_transp)*100:.4f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap='gray')
axes[1].imshow(opaque_aligned, cmap='gray')
axes[2].imshow(transp_aligned, cmap='gray')
plt.show()
