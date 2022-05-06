
# GPT

## Improving Language Understanding by Generative Pre-training

### Abstract

- 자연어 이해는 텍스트 수반, 질문 답변, 의미 유사성 평가 및 문서 분류와 같은 다양한 작업으로 구성됨
- 레이블이 지정되지 않은 큰 텍스트 말뭉치는 많이 존재하지만 모델을 훈련하기는 어려움.
- 이런 텍스트의 다양한 코퍼스에 대한 언어 모델의 생성적으로 사전 훈련된 것과 특정 작업에 대한 차별적 fine-tuning을 통해 이러한 작업에 대한 큰 이득을 실현할 수 있음.
- fine-tuning에서 작업 인식 입력 변환을 사용하면 모델 아키텍처에 대한 변경을 최소화하면서 효과적인 전달이 될 수 있다.
- 일반 작업 불가지론 모델은 다른 모델들에 비해 성능이 뛰어남

### 1. Introduction

레이블이 지정된 데이터를 구하기는 힘들며, 레이블이 지정되지 않은 데이터를 활용하기 위해서는 시간과 비용이 많이 소요될 수 있음

- 어떠한 objective를 이용해야 결과로 제공된 데이터를 transfer 해서 사용했을 때 효과적일지 명확하지 않음
- 학습된 representation을 target task에 어떠한 방식으로 transfer할지 명확한 방식이 정의되어 있지 않음

큰 규모의 레이블이 지정되지 않은 데이터와 task에 걸맞는 레이블이 지정된 데이터가 있을 때,

1. 레이블이 지정되지 않은 데이터로 모델의 초기 파라미터를 학습함
2. 초기화된 파라미터를 target task에 알맞게, 알맞는 objective에 맞게 추가적으로 학습

모델 구조로는 Transformer를 사용하며 멀리 떨어진 요소들 사이의 의존성을 학습하기에 적합함

### 2. Related Work

- Semi-supervised Learning for NLP
    
    레이블이 지정되지 않은 데이터로 학습한 단어 임베딩을 사용하는 방식이 제안됨
    
- Unsupervised Pre-Training
    - Semi-supervised Learning의 한 종류
    - 좋은 초기 환경을 제공함
    - LSTM 대신 Transformer 사용으로 보다 긴 의존성을 학습 가능
    - hidden representation을 사용
- Auxiliary Training objective
    
    보조적인 unsupervised training objective를 추가
    

### 3. Framework

- Unsupervised Pre-Training
    
    레이블이 지정되지 않은 데이터 U에 대해 likelihood를 최대화하는 방향으로 학습 진행
    
    ![Untitled 1](https://user-images.githubusercontent.com/104570858/167054408-c88a19a6-5edf-469e-bd69-9022c601f595.png)
    
    - 윈도우 크기 = k
    - θ 에 대해 conditional probability가 계산됨
    - SGD 방식으로 학습
    - multi-layer transformer decoder 사용
    - multi-headed self-attention연산을 모든 입력 토큰에 대해 수행하고, position-wise feedforward layer의 입력으로 제공함

![Untitled](https://user-images.githubusercontent.com/104570858/167054459-192beab5-1393-40cc-8a0d-0bb87532b830.png)

- Unsupervised fine-tuning
    
    supervised model의 generalization을 향상시킴과 동시에 빠르게 수렴 가능
    

![Untitled](https://user-images.githubusercontent.com/104570858/167054466-d6ba1cd5-67bd-41cb-ac34-0e4c06fb6d19.png)

![Untitled](https://user-images.githubusercontent.com/104570858/167054471-8450e8c7-3f98-4767-96a2-e100a264b7b3.png)

- 옴
    - Textual entailment : premise p와 hypothesis h를 delimiter token인 $로 구분
    - Similarity : similarity task의 경우, 두 입력 문장의 순서에 의미가 없기 때문에 모든 경우의 수를 고려해 element-wise addition을 통해 최종 representation을 생성
    - Question-answering, Commonsense reasoning : context document z와 question q, 답의 집힙 ak를 입력으로 제공해 representation을 생성 후 독립적으로 모델의 입력으로 제공되고 softmax 함수를 통해 결과를 예측하게 됨

### 4. Experiments

- Supervised fine-tuning

![Untitled](https://www.notion.so/GPT-77c4b108f3bc414e91d00b61a8061ecd#56dd108e3b8842aa8facbf613cf8cf17.png)

- Natural Language Inference (NLI)
    
    ![Untitled](https://user-images.githubusercontent.com/104570858/167054474-af8e7b98-a3a2-41c8-b1a7-286b26c8efb1.png)
    
- Question answering and commonsense reasoning

![Untitled](https://user-images.githubusercontent.com/104570858/167054480-0f9fd772-f626-4538-ad7f-7ee90df838a6.png)

- Semantic similarity
    - Classification

![Untitled](https://user-images.githubusercontent.com/104570858/167054484-4d3ee933-708a-4f57-9307-58e7ecd09c3.png)

### 5. Analysis

![Untitled](https://user-images.githubusercontent.com/104570858/167054488-56ab5c64-8ca7-4d01-a1aa-616ee4ce14ed.png)

- Impact of number of layers transferred
    - transfer하는 층의 개수가 많을수록 성능이 더 좋아짐
    - pre-trained model의 각 층이 target task를 해결하기 위한 다양한 특성들을 각각 학습함
- Zero-shot Behaviors
    
    NLP task를 위한 특성들을 함께 학습한다는 것을 알 수 있음
    

### 6. Conclusion

- 생성적 사전 훈련 및 판별적 미세 조정을 하는 프레임 워크 도입
- 긴 연속 텍스트가 있는 다양한 코퍼스에 대한 pre-training을 통해 최신 기술을 향상시킴
- 비지도 학습을 사용해 판별 작업의 성능을 높임
