## Midas 모델 사용법
1. 깃허브 클론 다운받기

    git clone https://github.com/isl-org/MiDaS.git
    cd MiDaS

2. 모델 다운로드

   1) <https://huggingface.co/Xiaodong/dpt_large_384/tree/8304ab53df28f5229c5e4665d6c5042d0eb4a088> 링크에 접속
   2) "dpt_large_384.pt" 파일 다운로드
   3) 다운로드 받은 파일을 Midas -> weights 폴더로 옮기기

3. 패키지 설치

    pip install torch torchvision timm opencv-python matplotlib
    pip install imutils

4. 이미지 넣기
   MiDas -> input 폴더에 원하는 이미지 넣기

5. MDE 실행

    python run.py --model_type dpt_large_384 --input_path input --output_path output

   
