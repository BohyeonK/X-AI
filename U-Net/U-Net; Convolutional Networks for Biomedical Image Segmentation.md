# U-Net

Created: 2022년 5월 8일 오전 12:26
Last Edited Time: 2022년 5월 13일 오후 3:25

# U-Net

## Abstract

Biomedical 분야에서 이미지 분할을 목적으로 제안됨

레이블이 있는 샘플을 효율적으로 사용하기 위해 **데이터 증강**에 의존하는 네트워크 및 훈련 전략을 제시함

아주 적은 수의 이미지로 End-to-End 방식의 FCN(Fully Convolutional Network) 기반으로 학습

- 아키텍처
    - 축소 경로 : 맥락 캡쳐의 역할
    - 확장 경로 : 정확한 위치 파악을 가능하게 하는 역할
- 특징
    - 네트워크가 빠름
    - 넓은 범위의 이미지 픽셀로부터 의미 정보를 추출하고 의미 정보를 기반으로 각 픽셀마다 객체를 분류하는 U 모양의 아키텍처
    - 서로 근접한 객체 경계를 잘 구분하도록 학습하기 위해 Weighted Loss를 제시

![Untitled](https://user-images.githubusercontent.com/104570858/168227430-09c1558e-b4e3-4971-abd0-e4fff574b002.png)

- FCN(Fully Convolutional Network) 구조를 응용한 구조를 사용함으로써 segmentation map을 출력
- Overlap-tile을 이용해 input image의 resolution이 GPU memory에 구애 받지 않고 사용 가능
- **Data Augmentation**을 이용해 적은 수의 이미지를 가지고도 적절한 학습이 가능

## Introduction

Biomedical image processing을 위해 localization 정보를 얻기 위해 sliding-window 방법 사용

- Sliding-Window
    - 기존 방식 : 한 칸씩 이동
    
  ![Untitled 1](https://user-images.githubusercontent.com/104570858/168227395-09b9a4db-d832-441b-af27-235bf758a65b.png)
    
    - U-Net의 방식 : 겹치지 않게 이동
    
    ![Untitled 2](https://user-images.githubusercontent.com/104570858/168227402-5f3dcba0-dfb3-4326-bbd0-ab8877d9f292.png)
    
    ![Untitled 3](https://user-images.githubusercontent.com/104570858/168227403-088baa3c-5eec-42e5-a6b9-b60f18edb5fb.png)
    
    - 기존 Sliding Window의 단점
        1. redundancy of over lapping patch
            - patch를 옮기면서 중복이 발생하게 됨
            - 중복된 부분은 학습된 부분을 다시 학습하게 되어 속도가 느려짐
        2. trade-off between localization accuracy and use of text
            - patch의 크기가 크면 max-pooling이 더 많이 적용됨
            - 정확한 위치 정보를 알기 어려워짐
            - 더 넓은 이미지를 보기 때문에 context 인식에는 효과를 가짐
        
        U-Net은 중복되지 않은 patch를 검증해 속도를 개선 시킴
        

## Network Architecture

- 개념
    - Contracting Path (수축 경로)
        - 크기가 줄어드는 부분
        - CNN 이미지의 context를 포착할 수 있도록 함
    - Expansive Path (확장 경로)
        - 크기가 증가되는 부분
        - 작아진 feature map을 upsampling해 원본 이미지와 비슷한 크기로 늘림
    - feature map과 upsampling한 feature map을 결합해 더 정확한 위치 정보를 가진 segmentation map을 얻을 수 있음
    - FCN과 유사하나 upsampling을 진행해도 높은 채널 수를 유지해 높은 layer 층에도 context 정보를 전파 가능
- Structure
    - 수축 경로
        - 각 층마다 2개의 Convolution layer 사용 (kernel size=3x3, non-padding, stride=1) → max-pooling(kernel size : 2x2, stride=2) 사용
        - 너비와 높이를 반으로 줄이고 channel의 크기를 2배로 늘림
        
        ![Untitled 4](https://user-images.githubusercontent.com/104570858/168227406-c9f9f5fa-e627-4a5a-809b-46176b7f1f6a.png)
        
    - 확장 경로 : 각 층마다 2개의 Convolution layer 사용 (kernel size=3x3, non-padding, stride=1) → up-convolution 사용
        - 분류 : 1x1 Convolution layer를 사용해 64개의 component feature vector를 desired number of class에 매핑하기 위해서 사용됨
        - feature map의 사이즈가 2배씩 커지기 때문에 (줄였던 것을 다시 늘려야 함) 채널의 수를 절반으로 줄여줌
- Overlap-tile
    
    ![Untitled 5](https://user-images.githubusercontent.com/104570858/168227410-270f3ad9-5054-4550-b66e-84cc98498f03.png)
    
    - 전자 현미경 데이터의 특성상 이미지 사이즈 크기가 매우 커 patch 단위로 잘라서 input으로 활용
    - padding을 사용하지 않기 때문에 이미지 크기가 줄어들게 됨
        
        → 전체 이미지의 크기와 같은 출력을 위해 가장자리는 이미지 반전을 사용함
        
        ![Untitled 6](https://user-images.githubusercontent.com/104570858/168227415-dba6bc4e-8060-4b23-8661-7da2bb668c02.png)
        

## Training

Input image & Segmentation Map은 SGD 기법과 함께 네트워크를 학습시키기 위해 사용됨

- Unpadded Convolution을 사용해 input image의 크기가 output image보다 큼
- 큰 batch_size보다는 큰 input tiles를 선호함
- 학습 진행시 단일 이미지의 input patch(tile)를 사용했고 batch_size를 작게 설정하기 때문에 momentum의 값을 0.99로 주어서 과거 Gradient Descent값을 더 많이 반영해 학습이 더 잘 되도록 함
- cross entropy loss
    
    본 논문에서는 cross entropy loss에 w(x)를 곱해서 사용
    
    ![Untitled 7](https://user-images.githubusercontent.com/104570858/168227417-0babe955-473e-4c78-9735-6156958f647c.png)
    
    - x는 feature map의 각 픽셀
    - 각 픽셀에서 계산한 것을 다 더해줌
    
    ![Untitled 8](https://user-images.githubusercontent.com/104570858/168227419-0ad19fe3-97f3-41d6-9233-fb6fec2ea563.png)
    
    - weight map : 특정 클래스가 가지는 픽셀의 주파수 차이를 보완해주는 식 (픽셀값 차이를 채워주는 역할)
    - 세포 사이에 떨어진 간격이 짧아 세포별로 구별이 힘듦
    - Touching cells separation
        - 세포 분할 작업에서 주요한 과제 중 하나는 동일한 클래스의 접촉 개체를 분리하는 것
        - 이미지 c,d 처럼 경계를 포착할 수 있어야 함
        - 이를 위해 학습 데이터에서 각 픽셀마다 클래스 분포가 다른 점을 고려해 사전에 groud-truth에 대한 weight map을 구해 학습에 반영했음
            
            ![Untitled 9](https://user-images.githubusercontent.com/104570858/168227420-89753b7a-f81a-45a4-9f96-f5665ff8e062.png)
            
        

### Data Augmentation

이미지 데이터를 회전, 자르기, 좌우반전, 밝기 조절 등의 방법으로 데이터의 수를 늘리는 방법

U-Net에서는 적은 데이터로 충분한 학습을 하기 위해 사용

(세포 데이터에서의 data augmentation은 좋은 효과를 가져옴)

## Experiments

EM segmentation challenge dataset으로 학습

- 512x512이미지 30장
- 세포는 흰색, 세포막은 검은색으로 색칠된 ground truth segmentation map
    
    ![Untitled 10](https://user-images.githubusercontent.com/104570858/168227425-d0cd7c9a-7906-4509-8993-68d9f5ce472f.png)
    
    # warping error : 객체 분할 및 병합이 잘 됐는지 세그멘테이션과 관련된 에러
    
    Warping Error에서 가장 좋은 성능을 보여줌
    
- 세포라고 구분한 영역과 실제 세포 영역이 겹치는 정도를 성능으로 측정
    
    ![Untitled 11](https://user-images.githubusercontent.com/104570858/168227427-54ad2753-f359-4980-929f-f2d9422f1e99.png)
    

## Conclusion

U-Net은 다양한 생물 의학 세분화 응용 프로그램에서 우수한 성능 달성

데이터 증강 기법 덕에 적은 데이터로도 활용 가능
