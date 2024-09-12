[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)
# Dialogue Summarization | 일상 대화 요약
## Team

| ![전은지](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이수형](https://avatars.githubusercontent.com/u/156163982?v=4) | ![서정민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이지윤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이승미](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [전은지](https://github.com/allisonej)             |            [이수형](https://github.com/dltngud1541)             |            [서정민](https://github.com/jmseo1216)             |            [이지윤](https://github.com/jiyuninha)             |            [이승미](https://github.com/seungmi1110)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

* **전은지(팀장)**: 팀원 서포터, 데이터 정제, 다중언어 번역
* **이승미(팀원)**: 데이터 분석 및 전처리, 모델 실험, 번역
* **서정민(팀원)**: 여러 모델 튜닝 및 실험
* **이지윤(팀원)**: 모델 튜닝 및 실험
* **이수형(팀원)**: 영문 번역, 모델 튜닝 및 실험

## 0. Setting
### 0.1 Environment
* *NVIDIA GeForce RTX 3090 24GB*  
* *AMD Ryzen Threadripper 3960X 24-Core Processor*  

### 0.2 Requirements
* *pandas*  
* *numpy*  
* *os*  
* *glob*  
* *torch*  
* *rouge*
* *pytorch_lightning*
* *transformers*
* *wandb*
* *nltk*
* *datasets*
* *huggingface_hub*

### 0.3 활용 장비 및 재료(개발 환경, 협업 tool 등)
* **(컴퓨팅 환경)**   
  * *NVIDIA GeForce RTX 3090 24GB*  
  * *AMD Ryzen Threadripper 3960X 24-Core Processor*  
* **(협업 환경)** *Github, Notion, Google Drive*
* **(의사 소통)** *Slack, Zoom*

## 1. Competiton Info

### 1.1 Overview

* **경진대회 주제:**  
본 프로젝트는 **일상 대화 데이터**를 기반으로 **대화 요약 모델**을 개발하는 경진대회의 일환으로 진행되었습니다. 목표는 다양한 주제의 대화문을 입력으로 받아 **효과적인 요약문**을 생성하는 모델을 구축하는 것입니다. 주요 성능 평가는 *ROUGE 지표(ROUGE-1, ROUGE-2, ROUGE-L)*를 사용하여 이루어졌습니다.

### 1.2 Timeline

* 대회 시작 : 2024.08.29 10:00 (목)  
* 데이터  분석 : 8/29\~8/30  
* 데이터 정제 : 8/30\~9/9  
  * 번역 : 8/30~  
  * 1~4차 : 8/30
  * 5차 : 9/1  
  * 6차 : 9/9  
* 모델 실험 : 8/30-9/6
* 성능 개선 : 9/6\~9/10
* 오프라인 2-3팀 해커톤 : 2024.09.08\~2024.09.09
* 대회 마감 : 2024.09.10 19:00 (화)

## 2. Data descrption

### 2.1 Dataset overview

* **Train 데이터셋**: 12,457개
	* train.csv
		* fname : 대화의 고유번호 (중복 없음)
        * dialogue : 2명에서 최대 7명이 등장하는 대화 내용. 개인정보와 이름은 특수 토큰(##)으로 처리
		* summary : 해당 대화를 바탕으로 작성된 요약문
		* topic : 대화문 주제


* **Validation 데이터셋**: 499개
	* dev.csv
		* fname : 대화의 고유번호 (중복 없음)
        * dialogue : 2명에서 최대 7명이 등장하는 대화 내용. 개인정보와 이름은 특수 토큰(##)으로 처리
		* summary : 해당 대화를 바탕으로 작성된 요약문
		* topic : 대화문 주제


* **Test 데이터셋**: 250개
	* test.csv
		* fname : 대화의 고유번호 (중복 없음)
        * dialogue : 2명에서 최대 7명이 등장하는 대화 내용. 개인정보와 이름은 특수 토큰(##)으로 처리


* **Hidden Test 데이터셋**: 249개
	* test.csv
		* fname : 대화의 고유번호 (중복 없음)
        * dialogue : 2명에서 최대 7명이 등장하는 대화 내용. 개인정보와 이름은 특수 토큰(##)으로 처리


### 2.2 Data Processing

#### 데이터 품질 개선
* **대화 내용 마스킹**: 이름, 번호 등 개인정보를 특수 토큰으로 마스킹 (#PhoneNumber, #Person1 등).
* **불필요한 기호 제거**: 기호 ‘#’, ‘:’, 번호 등 불필요한 문자 및 기호를 제거.
* **표기 통일성**: 스페이스와 기호 사용의 불일치를 수정해 통일성 유지 (예: [스페이스]/[스페이스] → /).
* **중복 표기 수정**: 동일 화자의 연속 발언 및 다른 규칙에 따른 표기 오류 수정.

#### 데이터 언어 번역
* **번역 작업**: 데이터는 기본적으로 한국어로 제공되었으나, 영어 및 일본어로의 번역을 시도하였습니다.
  * 영어 번역: **Googletran**s 및 **Solar API**를 사용.
  * 일본어 번역: **GPT** 및 **Solar API**를 사용.
* 번역 품질이 낮거나 번역 시 과도한 해석이 발생한 경우, 해당 부분을 재검토하여 수정하였습니다.
* 일본어의 경우 번역 오류가 많아 실제 학습에서는 사용하지 않았습니다.


## 3. Modeling

프로젝트에서는 다양한 모델 아키텍처와 프레임워크를 사용하여 실험을 진행했습니다. 모델들은 **Encoder-Decoder 구조**와 **Decoder Only 구조**로 구분됩니다.

### 3.1 (영문) Encoder-Decoder 구조
* Model: sshleifer/distilbart-cnn-12-6
<img width="422" alt="nlp_img1" src="https://github.com/user-attachments/assets/a6af310f-5c7d-447d-8e97-f4ac66618eac">

* Model: sshleifer/distilbart-xsum-12-3 <br>
<image2><img width="382" alt="nlp_img2" src="https://github.com/user-attachments/assets/4721dd20-33d4-4e72-9729-bf8d2e9f6a3d">

* Model: facebook/bart-large-xsum <br>
<image3><img width="406" alt="nlp_img3" src="https://github.com/user-attachments/assets/03244fa2-52e9-4e37-81cb-76f8a0ab8520">

* Model: google/flan-t5-large <br>
<image4><img width="373" alt="nlp_img4" src="https://github.com/user-attachments/assets/14700b34-b0c4-4118-a5f5-7ae8ca96266b">


### 3.2 (한글) Encoder-Decoder 구조
* Model 1: digit82/kobart-summarization (파라미터 수: 123,859,968)<br>
<image5><img width="551" alt="nlp_img5" src="https://github.com/user-attachments/assets/9961f6ee-7a62-4c00-b359-00d6de0776b2">

* Model 2: eenzeenee/t5-base-korean-summarization (파라미터 수: 236,904,192)<br>
<image6><img width="583" alt="nlp_img6" src="https://github.com/user-attachments/assets/7b327263-5b34-4435-b525-c6c530eaaee7">

* Model 3: lcw99/t5-large-korean-text-summary (파라미터 수: 768,918,528) - 최고 성능 <br>
<image7><img width="588" alt="nlp_img7" src="https://github.com/user-attachments/assets/76f60e83-a48b-4d59-b8da-03c0f751431a">

* Model 4: traintogpb/pko-t5-large-kor-for-colloquial-summarization-finetuned (LORA 사용) <br>
<image8><img width="591" alt="nlp_img8" src="https://github.com/user-attachments/assets/74dd91f1-e60c-437b-8416-9641c5012ca7">


### 3.3 Decoder Only 구조
* Model 1: cateto/korean-gpt-neox-125M (파라미터 수: 125,065,728)
* Model 2: MrBananaHuman/kogpt2_small (파라미터 수: 125,778,432)

### 3.4 LoRA 적용
* LoRA 모델: flan-t5-large
* 학습 조건:
  * learning_rate: 5e-4
  * train_epochs: 100
  * weight_decay: 0.015
  * batch_size: 1
  * lr_scheduler_type: 'linear'
* 비교 모델:
  * Original 모델: 기본 FLAN-T5
  * Instruct 모델: FLAN-T5를 Prompt 와 함께 파인튜닝한 버전
  * PEFT 모델: LoRA로 파인튜닝한 버전
* 성능 비교: LoRA를 적용한 모델은 Original 모델보다 성능이 향상되었고 메모리 상에서 효율성이 비교적 좋다는 것을 확인하였지만 굳이 LoRA를 사용할 이유는 되지 못하였습니다.
(성능 : Original < PEFT < Instruct model)
  * Original model <br>
  <image9><img width="451" alt="nlp_img9" src="https://github.com/user-attachments/assets/dd122bd4-d337-4fce-9fd8-dc97744f8ad4">

  * Instruct model <br>
  <image10><img width="485" alt="nlp_img10" src="https://github.com/user-attachments/assets/bb9bb768-5a7c-4421-866d-1da0a454e613">

  * PEFT model <br>
  <image11><img width="472" alt="nlp_img11" src="https://github.com/user-attachments/assets/d8a78e4e-6ba1-4ac5-a1f4-3b62b1f7660c">


### 3.5 주제별 모델 사용
대화의 다양한 주제에 맞춰 *8개의 카테고리*로 데이터를 나누어 각 카테고리에 특화된 모델을 사용하는 전략을 채택했습니다. 주제별(가족, 사회, 쇼핑, 업무, 엔터, 여행, 일상 생활, 주거)로 데이터를 나누어 학습한 모델은 다음과 같은 카테고리로 구분되었습니다:
* 사용한 주요 모델:
  * digit82/kobart-summarization
  * lcw99/t5-large-korean-text-summary
* 결과:
  * 데이터 부족때문인지 kobart 썼을때 학습결과가 많이 낮게 나왔습니다.
  * 또한 T5 모델의 경우 small 모델에서도 서버 메모리 초과 문제가 발생하여 실험을 중단하게 되었습니다.


## 4. 주요 도전 과제
* **메모리 초과 문제**: 큰 모델을 사용함에 따라 학습 도중 메모리 초과 및 서버 강제 종료 문제가 빈번히 발생하였습니다.
* **번역 품질 문제**: 번역 작업에서 과잉 해석이나 번역 오류가 발생하여 결과에 영향을 미쳤습니다.
* **결과의 불일치**: 생성형 모델의 특성상 시드를 고정하더라도 일관되지 않은 결과가 도출되는 경우가 있었습니다.

## 5. 성능 평가 및 결과
### 5.1 실험 결과
  * 성능이 좋은 모델:
    * lcw99/t5-large-korean-text-summary: 높은 성능을 기록했으며, 최종적으로 최고 성능 모델로 채택되었습니다.
    * LoRA 적용 모델은 성능 향상에 도움을 주었으나, GPU 리소스 사용이 많아 실험에 제약이 있었습니다.

## 6. 회고 및 느낀 점
### 6.1 팀 회고
* **서버 불안정성**: 서버가 자주 터지는 경험을 통해 서버 리소스 관리의 중요성을 실감하였습니다.
* **모델 학습의 한계**: 개인 PC에서 대형 모델을 학습하는 것에는 물리적 한계가 있음을 깨달았습니다.
* **NLP 학습 경험**: 처음으로 자연어 처리 프로젝트를 진행하면서, 자연어 데이터의 추상성을 이해하고 다양한 모델 실험을 통해 많은 경험을 쌓았습니다.

## 7. Result

### Leader Board

- 리더보드 [중간순위] <br>
<img width="518" alt="rank_mid" src="https://github.com/user-attachments/assets/3b77fa24-a826-4ca4-a0b3-fa04c9b12538">


- 리더보드 [최종순위] <br>
<img width="518" alt="rank_last" src="https://github.com/user-attachments/assets/ec0bcf16-02f7-4d5c-95ee-a139ed62b5b9">


- 최종 점수
  - Rank 6
  - Score : 41.4039



### Presentation
ppt link

## 8. etc

### Meeting Log

* Mon\~Fri 10:00\~13:00, 14:00\~19:00
* 2024-09-08 21:00\~2024-09-09 08:00  
* **Mentoring** : 2024-09-09 20:00\~21:00  

### Reference
* [*ref1*](https://www.kaggle.com/code/paultimothymooney/fine-tune-flan-t5-with-peft-lora-deeplearning-ai)
* [*ref2*](https://www.kaggle.com/code/lusfernandotorres/text-summarization-with-large-language-models)
* [*ref3*](https://www.kaggle.com/code/aisuko/fine-tuning-llm-for-dialogue-summarization)
