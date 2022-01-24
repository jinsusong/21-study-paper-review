# NLP paper review 

## 1. A Persona-Based Neural Conversation Model
    - 19 Mar 2016 

## 2. Google's Neural Machine Translation 
    - Nov 2016


## 3. Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation
    - 14 Nov 2016

## 4. Adversarial Examples for Evaluating Reading Comprehension Systems(EMNLP)
    - 23 Jul 2017

## 5. Unsupervised Machine Translation Using Monolingual Corpora Only(ICLR)
    - 31 Oct 2017

## 6. Graph Attention Networks(ICLR)
    - 30 Oct 2017

## 7. Universal Sentence Encoder(EMNLP)
    - 29 Mar 2018

## 8. What you can cram into a single vector: Probing sentence embeddings for linguistic properties(ACL)
    - 3 May 2018

## 9. Improving Language Understanding by Generative Pre-Training GPT1
    - Jun 2018
    
    1) 저자의 연구가 왜 중요한가? 
        - 원본 그대로의 텍스트 데이터를 효과적으로 학습하는 능력은 통해 NLP에서 지도학습에 대한 의존성을 낮춤
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Transformer : RNN의 문제점을 attention 기법으로 개선한 인코더 디코더 구조의 NLP 모델 , 병렬 계산 가능, 긴 문장 처리 가능
        - Self Attention : 문장 내 단어들 간의 연관성 포착
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        -  Transformer Decoder 만으로 약간의 fine-tuning 조정만으로 다양한 과제에 전이 및 적용할 수 있는 범용 표현 학습 모델을 구축함.
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - 넓은 범위에 걸친 언어적 정보를 포함하는 다양한 말뭉치에 대해 사전학습을 진행하여 전이되는 장거리 의존성을 처리하는 능력을 학습하여 
        일반지식과 질답, 의미유사성 평가, 함의 확인, 문서분류 등의 과제에 12개 중 9개의 과제에 대해 state-of-the-art를 달성
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 딥러닝 방법은 수동으로 분류된 방대한 양의 데이터를 필요로 하는데 이는 분류된 자원의 부족으로 인한 많은 범위로의 응용에 제약을 건다.    
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 미분류 데이터로부터 언어적 정보를 얻어낼 수 있는 모델을 구축해야함

## 10. Neural document summarization by jointly learning to score and select sentences (ACL)
    - 6 Jul 2018
    1) 저자의 연구가 왜 중요한가? 
        - 본 논문에서는 기존 두 단계로 이루어진 문장 채점과 문장 선택을 공동으로 학습하는 신경 추출 문서 요약(NEUSUM) 프레임워크를 제시함.
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - 이전 텍스트 요약 방식은 두 단계로 이루어짐 (문장 점수 매김 + 문장 선택)
            1) 단어 확률 기반 방법 TF-IDF
            2) 가중 그래프를 사용하는 그래프 기반 방법  TextRank ,LexRank
            3) 최근 신경망은 문장 모델링 및 채점에 활용됨 Cao , Ren
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - [문장 평가 + 문장 선택] = 하나로 결합된 end-to-end model
    
    4) 저자가 찾은 연구 결과가 무엇인가?
        - ROUGE 평가 결과는 제안된 공동 문장 점수 매기기 및 선택 접근법이 이전의 분리된 방법보다 훨씬 우수하다는 것
        - 제안된 모델은 최첨단 방법을 크게 능가하며 CNN/Daily Mail 데이터 세트에서 최상의 결과를 달성
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        -  문서 요약 시스템이 문장 채점과 문장 선택 두 개의 분리된 하위 작업으로 동작함.
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 이전 방법과 다르게 문장을 선택할 때마다 부분 출력 요약과 현재 추출 상태에 따라 문장 점수를 매기는 구조를 제안.

## 11. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    - 11 Oct 2018

## 12. Semi-Supervised Sequence Modeling With Cross-View Training(EMNLP)
    - 22 Sep 2018
    
    1) 저자의 연구가 왜 중요한가? 
        - 레이블이 없는 데이터를 활용하는 효과적인  Semi-Supervised training 기법을 개발.
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - self-training : NLP에 효과적이지만 신경 모델에서는 덜 사용하고 있음.
        - pre-training word vectors : NLP에서 활용중인 준지도 학습 전략
        - BiLSTM :최근 연구로는 Bi-LSTM 문장 인코더가 언어 모델링을 수행하도록 훈련시킨 다음 그 문맥에 민감한 표현을 supervised models에 통합함.
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 레이블링된 데이터와 레이블링되지 않은 데이터를 혼합하여 Bi-LSTM 문장 인코더의 표현을 개선하는 준지도 학습 알고리즘인 CVT(Cross-View Training)를 제안
    
    4) 저자가 찾은 연구 결과가 무엇인가?
        - 접근 방식을 통해 모델은 레이블이 없는 데이터에 대한 자체 예측을 효과적으로 활용하여 입력 중 일부를 사용할 수 없는 경우에도 정확한 예측을 산출하는 효과적인 표현을 생성하도록 훈련할 수 있음.
        - CVT가 다중 작업 학습과 결합될 때 7개의 NLP 작업에서 우수한 결과를 달성함.
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        -  Supervised 모델은 기본 교육 단계 동안에만 작업별 레이블링된 데이터로부터 학습함.
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - self-training과 같은 오래된 준지도 학습 알고리즘은 레이블링된 데이터와 레이블링되지 않은 데이터의 혼합에 대한 작업에 대해 지속적으로 학습하기 때문에 이 문제를 겪지 않는다.
        
## 13. A Survey on Deep Learning for Named Entity Recognition(IEEE TRANSACTIONS ON KNOWLEDGE ANDDATAENGINEERING)
    - 22 Dec 2018
    
    1) 저자의 연구가 왜 중요한가? 
        - 레이블이 없는 데이터를 활용하는 효과적인  Semi-Supervised training 기법을 개발.
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - BiLSTM :최근 연구로는 Bi-LSTM 문장 인코더가 언어 모델링을 수행하도록 훈련시킨 다음 그 문맥에 민감한 표현을 supervised models에 통합함.
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 
    
    4) 저자가 찾은 연구 결과가 무엇인가?
        - 
        - 
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 
        
## 14. Cross-lingual Language Model Pretraining
    - 22 Jan 2019

## 15. RoBERTa: A Robustly Optimized BERT Pretraining Approach
    - 26 Jul 2019

## 16. Text Summarization with Pretrained Encoders
    - 22 Aug 2019

## 17. Language Models are Unsupervised Multitask Learners GPT2 
    - Aug 2019
 
    1) 저자의 연구가 왜 중요한가? 
        - 
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - 
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        -
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        -
        
## 18. ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
    - 23 Mar 2020
