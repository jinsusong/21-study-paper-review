# NLP paper review 
1. [A Persona-Based Neural Conversation Model](#1-a-persona-based-neural-conversation-model)
2. [Google's Neural Machine Translation](#2-google-neural-machine-translation)
3. [Google Multilingual Neural Machine Translation System Enabling Zero Shot Translation](#3-google-multilingual-neural-machine-translation-system-enabling-zero-shot-translation)
4. [Adversarial Examples for Evaluating Reading Comprehension Systems(EMNLP)](#4-adversarial-examples-for-evaluating-reading-comprehension-systems)
5. [Unsupervised Machine Translation Using Monolingual Corpora Only(ICLR)](#5-unsupervised-machine-translation-using-monolingual-corpora-only)
6. [Graph Attention Networks(ICLR)](#6-graph-attention-networks)
7. [Universal Sentence Encoder(EMNLP)](#7-universal-sentence-encoder)
8. [What you can cram into a single vector: Probing sentence embeddings for linguistic properties(ACL)](#8-what-you-can-cram-into-a-single-vector-probing-sentence-embeddings-for-linguistic-properties)
9. [Improving Language Understanding by Generative Pre-Training GPT1](#9-improving-language-understanding-by-generative-pre-training)
10. [Neural document summarization by jointly learning to score and select sentences (ACL)](#10-Neural-document-summarization-by-jointly-learning-to-score-and-select-sentences)
11. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](#11-BERT-pre-training-of-deep-bidirectional-transformers-for-language-understanding)
12. [Semi-Supervised Sequence Modeling With Cross-View Training(EMNLP)](#12-semi-supervised-sequence-modeling-with-cross-view-training)
13. [A Survey on Deep Learning for Named Entity Recognition(IEEE TRANSACTIONS ON KNOWLEDGE ANDDATAENGINEERING)](#13-a-survey-on-deep-learning-for-named-entity-recognition)
14. [Cross-lingual Language Model Pretraining](#14-cross-lingual-language-model-pretraining)
15. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](#15-roberta-a-robustly-optimized-bert-pretraining-approach)
16. [Text Summarization with Pretrained Encoders](#16-text-summarization-with-pretrained-encoders)
17. [Language Models are Unsupervised Multitask Learners GPT2 ](#17-language-models-are-unsupervised-multitask-learners-gpt2)
18. [ELECTRA Pre-training Text Encoders as Discriminators Rather Than Generators](#18-electra-pre-training-text-encoders-as-discriminators-rather-than-generators)

# 1 A Persona-Based Neural Conversation Model
    - 19 Mar 2016 
    
    1) 저자의 연구가 왜 중요한가? 
        - 화자의 페르소나를 고려한 응답생성 모델을 구축함
            1) Speaker Model
                - 응답을 내놓게 되는 화자의 화자벡터를 디코더의 인풋과 함께 넣어주는 모델
                - 화자에 맞는 적합한 응답을 출력할 수 있도록 목표함
            2) Speaker-Addressee Model
                - 각 화자가 상대방에 따라 발화한 문장들을 바탕으로 화자를 특징짓는 모델
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Data-driven response generation in social media : 대화의 생성 문제를 통계적 기계 번역 문제로 취급한 것 , 사람의 개입없이 학습함
        - A neural network approach to context-sensitive generation of conversational responses : Ritter et al의 확장 연구 , continuous RLM을 사용해 과거의 대화 내용을 기반으로 다음 대답을 생성하는 방법
        - Building end-to-end dialogue systems using generative hierarchical neural network models : 확장된 대화 기록에 대한 종속성을 파악하기 위해 hierarchical neural model 을 제안
        - All the World's a Stage : Learning Character Models from Film : 영화스크립트를 기반으로 캐릭터 유형에 따른 모델을 학습
        - 최근 관련 연구 : 논문을 인용한 papers 데이터 셋의 구축을 통해 페르소나 문제를 해결해보자
            1. Personalizinf Dialogue Agents : Crowd-sourcing으로 페르소나 데이터셋 구축
            2. Training Millions of Personalized Dialogue Agents : Crowd-sourcing은 부족, Reddit 데이터를 이용해서 대량의 페르소나 데이터셋 구축
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - Persona모델이 개인적 특성을 포착할 수 있음을 보여줌.
        - Speaker-Addressee모델에서는 이원적 특성을 포착할 수 있음을 보여줌.
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - BLEU, Perplexity, 인적 평가에서 Speaker의 Consistency는 Baseline보다 뛰어남.
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 오픈도메인에서 자연스러운 대화를 위한 연구를 진행되면서 문제점이 발견됨
        - 데이터 기반 시스템의 문제 : training data 중 가장 가능성이 높은 응답을 선택
            1. 결과가 모호함
            2. 일관성이 없음
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - consistency : 일관성문제 
        - coherent persona : 일관된 페르소나를 부여하는 방법

# 2 Google Neural Machine Translation 
    - Nov 2016
    
    1) 저자의 연구가 왜 중요한가? 
        - Neural Machine Translation 방식
            1. End-to-end learning 방식을 사용
            2. 간단한 구조를 가지고 있음.
            3. 입력된 문장 전부를 보고 번역을 할 수 있다는 장점이 있음.  
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Seq2Seq Model : 굉장히 긴 sequence가 들어온다면 고정된 컨텍스트 벡터로 인해 데이터 유실이 발생됨
        - Attention Models : Seq2Seq의 Larger vocabulary 한계를 극복하여 성능을 개선한 Paper
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
         - 제안 : seq2seq + Bahanau attention + Residual connction
                 
    4) 저자가 찾은 연구 결과가 무엇인가?
        - BLUE score 는 많이 올랐지만 
        - Human 평가에서는 성능이 좋지 않았다
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 16년 이전에는 통계방식으로 혹은 룰 베이스로 기계번역 NLP를 처리함.
        - Phrase-based translation 을 많이 사용함 (묶이는 구와 절 단위가 있을 것이다. 그걸 묶어서 번역을 해야한다. 라는 아이디어로 진행하는 아이디어)
        - 번역은 입력문장 전체를 입력단위를 두고 번역을 해야 정확한 문법 요소를 파악할 수 있음
        - 기존의 방식은 구 또는 단어 단위로 독립적으로 번역하기 때문에 정확한 번역을 하기 어렵다
        - 어떻게 분리할 것인지도 선택해야하기 때문에 성능을 내기도 어렵다.
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 느린 학습/추론 속도
        - 비효율적인 rare words 처리 ( 학습 할 때 몰랐던 단어 )
        - Source 문장 coverage

# 3 Google Multilingual Neural Machine Translation System Enabling Zero Shot Translation
    - 14 Nov 2016
    
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

# 4 Adversarial Examples for Evaluating Reading Comprehension Systems
    - 23 Jul 2017
    
    1) 저자의 연구가 왜 중요한가? 
        - 질문과 비슷한 문장을 생성하여 문단의 제일 끝에 추가하고 언어모델이 추가된 가짜 질문에 속지 않고 기존 전체 내용에서 올바른 답변을 추출하는지 확인
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Standord Question Answering Dataset (SQuAD)
            1. 위키피디아 글에 대해 사람이 생성한 Question으로 구성된 데이터 셋
            2. 데이터 셋 하나마다 Paragraph + Question + Answer로 구성
        - Models
            1. BiDAF(Bidirectional Attention Flow)
            2. LSTM
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 네 가지 모델을 가지고 AddSent , AddOneSent, AddAny, AddCommon의 적대적 예시구문을 추가해서 테스트한 결과
        - 모델들이 adversary에 대처를 잘하는 모습을 보이지 않음.
        - addAny가 AddSent보다 더 효과적으로 성능을 떨어뜨림.
        - 해당 작업이 언어를 더 깊이 이해하는 보다 정교한 모델의 개발에 동기를 부여하기를 바람
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - 표준 평가 지표에 따른 모델의 성능은 성공적이었음에도 불구하고, 
        - 기존의 reading Comprehension System은 adversarial evaluation에서 성능이 떨어짐
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 많은 Reading Comprehension System들의 성능이 빠르게 향상되고 있지만, 이 모델들이 실제로 언어를 이해하고 있는 것인지는 확실치 않음
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 실제로 언어를 이해하는지 확인하기 위해 문단에 Adversarial Sentences가 추가되었을 때에도 질문에 대한 답변을 잘 하는지를 확인하기 위한 데이터셋을 생성

# 5 Unsupervised Machine Translation Using Monolingual Corpora Only
    - 31 Oct 2017
    
    1) 저자의 연구가 왜 중요한가? 
        - 적은 parallel sentence로 학습을 하는것보다 단일 언어로 이루어진 많은 데이터를 학습하는것이 더 좋다
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Adversarial Attack : 이미 훈련된 모델에 대해 입력 데이터를 조작해 잘못 예측하도록 함
        - Auto Encoder
            1. 출력이 입력 데이터와 같아지도록 학습한 네트워크
            2. 차원 축소, noise 제거, 이상 데이터 검출, pre-train 등에 활용
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - lower resource한 언어들이나 semi supervied를 활용해 더 나은 기계번역 방법이 나올 수 있음을 기대함.
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - Parallel dataset이 아니기 때문에 번역의 품질 평가 어려움
        - 따라서 input을 2step 번역을 통해 재구성하여 재구성한 문장과 input을 비교
        - 제시한 모델은 한번만 반복해도 기존 방식보다 더 나은 성능을 확인할 수 있었음.
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 일반적으로 기계번역은 source sentence와 target sentence가 쌍을 이루는 parallel 데이터가 필요함.
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - source sentence와 target sentence가 쌍을 이루는 parallel는 구하기도 힘들고 구축하는데 많은 비용이 들어감

# 6 Graph Attention Networks
    - 30 Oct 2017
    
    1) 저자의 연구가 왜 중요한가? 
        - GAT는 어떤한 노드의 embedding을 생성할 때 인접한 노드들에 대한 중요도를 계산하여 이를 기반으로 새로운 embedding을 만듦
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - CNN
            많은 layer가 필요함,
            자연어의 길이가 너무 길어지면 관계를 알 수 없게됨.
        - RNN
           구조 상 순차적으로 학습이 진행되기 때문에 병렬연산이 되지 않음.
            시간복잡도가 올라가는 단점이 있음,
            Vanishing gradient problem(기울기 소멸 문제) : 신경망의 활성함수의 도함수값이 계속 곱해지다보면 가중치에 따른 결과값의 기울기가 0이 되어 버려서, 경사 하강법을 이용할 수 없게 되는 문제
            Long Term dependency(장기 의존성) :
        - Attention
            계산 복잡도를 줄이고
            병렬화를 통해 길이가 길어지면서 발생하는 문제를 해결할 수 있음.
        - About Graph
            Graph
            Node
            Edge
            Adjacency matrix
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - GCN과의 비교를 통해, 같은 neighborthood 내의 node 들에 대해 다른 weight를 부여하는 방법이 효과적임.
            
    4) 저자가 찾은 연구 결과가 무엇인가?
        - GAT는 graph data structure에 적용될 수 있는 convolution을 제안한 것으로 다음과 같은 특징을 갖는다.
            1. 장점 (효율적인 연산, 노드별 중요도 부여, Inductive learning 가능)
                Self attention layer를 통해 node들마다 병렬화가 가능하며, 이웃 노드별로 다른 중요도를 부과할 수 있다.
            2. 단점
                각기 다른 이웃 Node에 대하여 attention score를 구함으로 계산 효율성을 떨어진다. (PPNP논문 비교 결과)  
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 가장 대표적인 GNN인 graph convolutional network (GCN)에서는 노드 embedding layer가 graph convolution이라는 연산으로 정의됨.
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - GAT에서는 self-attention으로 노드 embedding layer을 정의함.

# 7 Universal Sentence Encoder
    - 29 Mar 2018
    
    1) 저자의 연구가 왜 중요한가? 
        - 문장 embedding을 활용한 전이학습이 단어 수준의 전이학습보다 뛰어난 결과를 보여주고, 적은 양의 지도 학습 데이터로도 좋은 성능을 나타냄.
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Word embedding
            1. Word2vec : CBOW 모델(주변 단어로부터 목표 단어 예측)과 Skip-gram 모델(중심 단어로부터 주변 단어 예측)
            2. GloVe : 
                - Word2vec : 주변 단어 중심으로 학습함으로써 말뭉치의 전체적인 통계 정보를 반영하지 못하는 문제가 존재
                - 중심 단어와 주변 단어의 내적이 전체 말뭉치에서의 동시 발생 확률이 되도록 하는 목적 함수 설정
        - Sentence Embedding
            1. Doc2Vec : Word2vec에서 확장된 개념으로, 문장-문단-문서 단위로 vector 계산
            2. InferSent : entailment/contradiction/neutral로 라벨링된 영어 자연어 데이터로 지도 학습된 모델
        - Skip -Thoughts
            1. 고정된 길이로 문장을 표현하는 비지도 방식의 신경망 모델
            2. 자연어 말뭉치에서 문장의 순서를 정보로 사용
            
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 결과적으로 워드 임베딩보다 문장 임베딩이 좋은 성능을 보였고 두 개를 같이 사용하는 것이 가장 효과가 좋았다.
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - sentence embedding을 사용해, 매우 적은 학습 데이터로도 좋은 성능을 보임.
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - 제한적인 NLP 학습 데이터, 비용이 많이 드는 annotation 작업
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - word2vec, GloVe와 같은 pre-trained word embeddings를 사용해 한계를 극복하려는 시도

# 8 What you can cram into a single vector Probing sentence embeddings for linguistic properties
    - 3 May 2018
    
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

# 9 Improving Language Understanding by Generative Pre Training
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

# 10 Neural document summarization by jointly learning to score and select sentences
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

# 11 BERT Pre training of Deep Bidirectional Transformers for Language Understanding
    - 11 Oct 2018

# 12 Semi Supervised Sequence Modeling With Cross View Training
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
        
# 13 A Survey on Deep Learning for Named Entity Recognition
    - 22 Dec 2018
    
    1) 저자의 연구가 왜 중요한가? 
        - 딥러닝 기반 NER 솔루션에 대한 최근 연구를 검토하여 새로운 연구자들이 이 분야에 대한 종합적인 이해를 쌓을 수 있도록 돕는 것을 목적
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - Rule-based approaches :
            1)domain-specific 사전, 구문-어휘 패턴을 기반으로 규칙을 설계
            2)도메인 고유 규칙과 불완전한 사전으로 높은 정밀도, 낮은 재현율 관찰
            3)다른 도메인으로 시스템을 이전 X
        - Unsupervised learning :
            1) 가장 대표적인 접근법은 클러스터링
            2) 컨텍스트 유사성을 기반으로 클러스터링된 그룹에서 NE를 추출
            3)대규모 코퍼스에서 계산된 통계를 사용해 NE 언급 추론
        - Feature-based supervised learning
            1) 지도학습을 적용하면 NER은 다중 클래스 분류나 시퀀스 레이블 지정 테스크가 됨.
            2) 어노테이션된 데이터 샘플이 주어지면 feature들이 각 training 예시를 잘 나타내도록 설계됨
            3) 이후 기계학습 알고리즘을 사용해 데이터에서 유사함 패턴을 인식하는 모델을 학습함. 
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        - 이 설문조사가 DL 기반 NER 모델을 설계할 때 좋은 참고가 될 수 있기를 바람
    
    4) 저자가 찾은 연구 결과가 무엇인가?
        - Dee-learning 을 적용하는 이유
            1) 활성화함수 덕분에 입력에서 출력으로 비선형 변환이 쉽다.
            2) NER 기능의 설계를 간단하게 해줌.
            3) end-to-end 학습이 가능함.
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - NER 연구는 수십 년 동안 번창해왔지만, 이 분야에서는 지금까지 DL 리뷰가 거의 없다. 
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - Data annotation :
            1) 많은 데이터를 위한 시간과 비용의 문제
            2) 주석 작업을 수행하기 위해서는 전문가가 필요함. 아직 자원이 부족한 많은 언어와 특정 도메인이 많은 문제
            3) 언어의 모호성으로 인한 주석의 품질과 일관성의 문제
            4) 하나의 명명된 엔터티에 여러 유형이 할당될 수 있는 중첩 엔터티와 세분화된 엔터티 모두에 적용할 수 있는 공통 주석 체계 개발이 필요함
        - Information Text and Unseen Entities :
            1) 신조어, 비공식 텍스트에 대한 낮은 정확도
            2) 이전에 볼 수 없었던 비형식적 텍스트를 식별하는 능력 필요
        
# 14 Cross-lingual Language Model Pretraining
    - 22 Jan 2019
    
    1) 저자의 연구가 왜 중요한가? 
        - 영어 자연어 이해를 위한 pre-training의 효율성을 다국어로 확장함.
        
    2) 기존에 어떤 연구들이 이루어졌는가?
        - 문장 인코더의 생성적 pre-training
        - 트랜스포머 언어 모델은 대규모 비지도 텍스트 말뭉치에서 학습된 다음 분류 또는 자연어 추론과 같은 자연어 이해 작업에 대해 fine-tuning함.
        
    3) 저자의 연구가 이 분야에 무엇을 기여하는가?
        1. 새로운 비지도 학습 방법 제시
        2. 병렬데이터를 이용할 수 있는 지도 학습 방법 제시
        3. 기계번역에서의 성능평가
        4. Low resource 언어의 복잡도 개선 
        
    4) 저자가 찾은 연구 결과가 무엇인가?
        - 교차 언어 분류, 감독되지 않은 기계 번역 및 감독된 기계 번역에서 이전의 최신 기술을 크게 능가
        - 교차 언어 언어 모델이 저자원 언어의 복잡성을 크게 개선할 수 있다는 것을 보여줌
        
    5) 저자의 연구가 제기하는 문제는 무엇인가?
        - Bert와 같은 기존의 pre-training model 들이 좋은 성능을 보였지만, 대부분 영어를 중심으로 연구가 진행됨
        
    6) 연구 문제를 위해 풀어야 하는 구체적인 문제는 무엇인가? 
        - 다양한 나라의 언어가 하나의 임베딩 공간을 공유하여, 어떤 문장이라도 해당 임베딩 공간에 인코딩할 수 있는 Universal cross-lingual encoder 를 만들어야함.

# 15 RoBERTa A Robustly Optimized BERT Pretraining Approach
    - 26 Jul 2019
    
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

# 16 Text Summarization with Pretrained Encoders
    - 22 Aug 2019
    
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

# 17 Language Models are Unsupervised Multitask Learners GPT2
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
        
# 18 ELECTRA Pre-training Text Encoders as Discriminators Rather Than Generators
    - 23 Mar 2020
    
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
    
