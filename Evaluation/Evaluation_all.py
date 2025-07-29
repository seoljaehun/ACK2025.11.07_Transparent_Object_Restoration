# 불투명 VS 투명 물체의 MDE 성능 비교(all)
# 투명 물체 부분만 정규화

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# .exr 이미지 파일을 openCV에 업로드하기위해 필요한 명령어
import cv2
import numpy as np

""" PFM 파일을 읽어 numpy float32 배열과 스케일 반환. """
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
    gt_safe = np.clip(gt, 1e-8, None)
    return np.mean(np.abs(gt - pred) / gt_safe)

# δ accuracy: 상대 오차 허용 기준 내 비율 (클수록 좋음)
def delta_accuracy(gt, pred, threshold=1.05): # 기준값 = 1.05
    ratio = np.maximum(gt / pred, pred / gt)
    return np.mean(ratio < threshold)

base_dir = r"C:\SJH\Python\Transparent_Object_Detection\Data\Opaque_Transparent_Comparison"  # Data 폴더 경로
gt_dir = os.path.join(base_dir, "GT_image") # 정답 이미지(GT) 폴더 경로
mask_dir = os.path.join(base_dir, "Mask_image") # 투명 물체 Mask 이미지 폴더 경로
opaque_dir = os.path.join(base_dir, "Opaque_image") # 불투명 물체 뎁스 이미지 폴더 경로
transp_dir = os.path.join(base_dir, "Transparent_image") # 투명 물체 뎁스 이미지 폴더 경로

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
    opaque_path = os.path.join(opaque_dir, f"{file_id}-opaque-rgb-img-dpt_large_384.pfm") # 불투명 물체 뎁스 이미지 파일 경로
    transp_path = os.path.join(transp_dir, f"{file_id}-transparent-rgb-img-dpt_large_384.pfm") # 투명 물체 뎁스 이미지 파일 경로

    # 경로가 존재하기 않으면 종료
    if not all(map(os.path.exists, [gt_path, mask_path, opaque_path, transp_path])):
        continue
    
    try:
        # 이미지 업로드
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        opaque, scale_A = load_pfm(opaque_path)
        transp, scale_B = load_pfm(transp_path)
        
        # GT 이미지를 Mask 이미지와 같은 size로 변경
        target_size = (mask.shape[1], mask.shape[0])
        gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Mask 이미지에서 이진 분류, 투명 물체 부분만 1값 나머지 0값
        binary_mask = (mask == 255).astype(np.uint8)
        
        # 투명 물체 Mask 적용
        opaque_object = opaque[binary_mask == 1]
        gt_object = gt[binary_mask == 1]

        # 물체 영역에서 유효값 필터링
        valid = (gt_object > 0) & (opaque_object > 0)
        p = opaque_object[valid]        # 예측 상대적인 disparity
        g = gt_object[valid]            # GT

        # Scale + Shift 정규화
        # 최소 제곱법으로 scale (s), shift (t) 계산,, 불투명 이미지 기준으로 계산
        A = np.vstack([p, np.ones_like(p)]).T
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        
        # 예측값 보정 및 역수
        opaque_aligned = np.clip(s * opaque + t, 1e-8, None)
        transp_aligned = np.clip(s * transp + t, 1e-8, None)
        
        # 각 이미지에 Mask 적용
        gt_masked = gt[binary_mask == 1]
        opaque_masked = opaque_aligned[binary_mask == 1]
        transp_masked = transp_aligned[binary_mask == 1]
        
        # 마스킹 영역에서 유효한 값만 필터링
        valid = (gt_masked > 0) & (opaque_masked > 0) & (transp_masked > 0)
        g = gt_masked[valid]
        p_opaque = opaque_masked[valid]
        p_transp = transp_masked[valid]
        
        # 평가지표 계산 후 누적 변수에 추가
        # 불투명 이미지 평가지표
        sum_opaque['rmse'] += rmse(g, p_opaque)
        sum_opaque['mae'] += mae(g, p_opaque)
        sum_opaque['rel'] += abs_rel(g, p_opaque)
        sum_opaque['delta'] += delta_accuracy(g, p_opaque)
        # 투명 이미지 평가지표
        sum_transp['rmse'] += rmse(g, p_transp)
        sum_transp['mae'] += mae(g, p_transp)
        sum_transp['rel'] += abs_rel(g, p_transp)
        sum_transp['delta'] += delta_accuracy(g, p_transp)
        
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
    print("유효한 이미지가 없습니다.")
