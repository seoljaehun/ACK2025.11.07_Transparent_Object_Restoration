# 불투명 VS 투명 물체의 MDE 성능 비교

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# .exr 이미지 파일을 openCV에 업로드하기위해 필요한 명령어
import cv2
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"C:\VScode_Data"  # Data 폴더 경로
gt_dir = os.path.join(base_dir, "GT_image") # 정답 이미지(GT) 폴더 경로
mask_dir = os.path.join(base_dir, "Mask_Depth_image") # 투명 물체 Mask 이미지 폴더 경로
opaque_dir = os.path.join(base_dir, "Opaque_Depth_image") # 불투명 물체 뎁스 이미지 폴더 경로
transp_dir = os.path.join(base_dir, "Transparent_Depth_image") # 투명 물체 뎁스 이미지 폴더 경로

# 총 4가지의 평가지표 정의
# RMSE: 루트 평균 오차 제곱 (낮을수록 좋음)
def rmse(gt, pred): 
    return np.sqrt(np.mean((gt - pred) ** 2))

# MAE: 평균 절대 오차 (낮을수록 좋음)
def mae(gt, pred): 
    return np.mean(np.abs(gt - pred))

# abs REL: 절대 상대오차 (낮을수록 좋음)
def abs_rel(gt, pred): 
    gt_safe = np.clip(gt, 1e-8, None)
    return np.mean(np.abs(gt - pred) / gt_safe)

# δ accuracy: 상대 오차 허용 기준 내 비율 (클수록 좋음)
def delta_accuracy(gt, pred, threshold=1.15): # 기준값 = 1.15
    ratio = np.maximum(gt / pred, pred / gt)
    return np.mean(ratio < threshold)

# 누적 변수 초기화
sum_opaque = {'rmse': 0, 'mae': 0, 'rel': 0, 'delta': 0} # 불투명물체 평가지표 변수
sum_transp = {'rmse': 0, 'mae': 0, 'rel': 0, 'delta': 0} # 투명 물체 평가지표 변수
valid_count = 0 # 이미지 비교 횟수

# 파일 이름 불러오기
for fname in sorted(os.listdir(gt_dir)):
    if not fname.endswith(".exr"): # .exr 파일만 처리
        continue
    file_id = fname.split("-")[0]  # 예: '000000001'
    
    gt_path = os.path.join(gt_dir, f"{file_id}-output-depth.exr") # 정답 이미지(GT) 파일 경로
    mask_path = os.path.join(mask_dir, f"{file_id}-mask.png") # 투명 물체 Mask 이미지 파일 경로
    opaque_path = os.path.join(opaque_dir, f"{file_id}-opaque-rgb-img-dpt_large_384.png") # 불투명 물체 뎁스 이미지 파일 경로
    transp_path = os.path.join(transp_dir, f"{file_id}-transparent-rgb-img-dpt_large_384.png") # 투명 물체 뎁스 이미지 파일 경로

    # 경로가 존재하기 않으면 종료
    if not all(map(os.path.exists, [gt_path, mask_path, opaque_path, transp_path])):
        continue
    
    try:
        # 이미지 업로드(1차원, float32)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        opaque = cv2.imread(opaque_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
        transp = cv2.imread(transp_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
        
        # GT 이미지를 Mask 이미지와 같은 size로 변경
        target_size = (mask.shape[1], mask.shape[0])
        gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Mask 이미지에서 이진 분류, 투명 물체 부분만 1값 나머지 0값
        binary_mask = (mask == 255).astype(np.uint8)
        
        # 각 이미지에 Mask 적용
        gt_masked = gt[binary_mask == 1]
        opaque_masked = opaque[binary_mask == 1]
        transp_masked = transp[binary_mask == 1]
        
        # 유효한 값만 필터링
        valid = (gt_masked > 0) & (opaque_masked > 0) & (transp_masked > 0)
        if np.sum(valid) == 0:
            continue
        g = gt_masked[valid]
        p_opaque = opaque_masked[valid]
        p_transp = transp_masked[valid]
        
        # Scale + Shift 정규화
        # 최소 제곱법으로 scale (s), shift (t) 계산,, 불투명 이미지 기준으로 계산
        A = np.vstack([p_opaque, np.ones_like(p_opaque)]).T
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        
        # 예측값 보정
        p_opaque_aligned = np.clip(s * p_opaque + t, 0, None)
        p_transp_aligned = np.clip(s * p_transp + t, 0, None)
        
        # 평가지표 계산 후 누적 변수에 추가
        # 불투명 이미지 평가지표
        sum_opaque['rmse'] += rmse(g, p_opaque_aligned)
        sum_opaque['mae'] += mae(g, p_opaque_aligned)
        sum_opaque['rel'] += abs_rel(g, p_opaque_aligned)
        sum_opaque['delta'] += delta_accuracy(g, p_opaque_aligned)
        # 투명 이미지 평가지표
        sum_transp['rmse'] += rmse(g, p_transp_aligned)
        sum_transp['mae'] += mae(g, p_transp_aligned)
        sum_transp['rel'] += abs_rel(g, p_transp_aligned)
        sum_transp['delta'] += delta_accuracy(g, p_transp_aligned)
        
        # 횟수 카운터
        valid_count += 1
    
    # 예외    
    except Exception as e:
        print(f"[{file_id}] 에러 발생: {e}")
        continue
    
# 평가지표 출력
if valid_count > 0:
    print(f"\n평가 완료: 총 {valid_count} 쌍 처리됨")
    # 불투명 물체의 평가지표를 합하고 횟수로 나누어 평균 구하기
    for k in sum_opaque:
        print(f"{k.upper()} (불투명): {sum_opaque[k] / valid_count:.4f}")
    # 투명 물체의 평가지표를 합하고 횟수로 나누어 평균 구하기
    for k in sum_transp:
        print(f"{k.upper()} (투명):   {sum_transp[k] / valid_count:.4f}")
else:
    print("❌ 유효한 이미지가 없습니다.")