# -*- coding: utf-8 -*-
"""Combined Topic Model을 이용한 토픽 모델링(한국어)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X0CtRGsXUtxfE5flMkHG6RJOgVgc_wT4

# 복합 토픽 모델링(Combined Topic Modeling)

이 튜토리얼에서는 복합 토픽 모델(**Combined Topic Model**)을 사용하여 문서의 집합에서 토픽을 추출해보겠습니다.

## 토픽 모델(Topic Models)

토픽 모델을 사용하면 비지도 학습 방식으로 문서에 잠재된 토픽을 추출할 수 있습니다.

## 문맥을 반영한 토픽 모델(Contextualized Topic Models)
문맥을 반영한 토픽 모델(Contextualized Topic Models, CTM)이란 무엇일까요? CTM은 BERT 임베딩의 표현력과 토픽 모델의 비지도 학습의 능력을 결합하여 문서에서 주제를 가져오는 토픽 모델의 일종입니다.


## 파이썬 패키지(Python Package)

패키지는 여기를 참고하세요 [링크](https://github.com/MilaNLProc/contextualized-topic-models).

![https://github.com/MilaNLProc/contextualized-topic-models/actions](https://github.com/MilaNLProc/contextualized-topic-models/workflows/Python%20package/badge.svg) ![https://pypi.python.org/pypi/contextualized_topic_models](https://img.shields.io/pypi/v/contextualized_topic_models.svg) ![https://pepy.tech/badge/contextualized-topic-models](https://pepy.tech/badge/contextualized-topic-models)

# **시작하기 전에...**

이 튜토리얼과 관련하여 추가적인 의문 사항이 있다면 아래 링크를 참고하시기 바랍니다:

- 영어가 아닌 다른 언어로 작업하고 싶으시다면: [여기를 클릭!](https://contextualized-topic-models.readthedocs.io/en/latest/language.html#language-specific)
- 토픽 모델에서 좋은 결과가 나오지 않는다면: [여기를 클릭!](https://contextualized-topic-models.readthedocs.io/en/latest/faq.html#i-am-getting-very-poor-results-what-can-i-do)
- 여러분의 임베딩을 사용하고 싶다면: [여기를 클릭!](https://contextualized-topic-models.readthedocs.io/en/latest/faq.html#can-i-load-my-own-embeddings)

# GPU를 사용하세요

우선, Colab에서 실습하기 전에 GPU 설정을 해주세요:

- 런타임 > 런타임 유형 변경을 클릭하세요.
- 노트 설정 > 하드웨어 가속기에서 'GPU'를 선택해주세요.

[Reference](https://colab.research.google.com/notebooks/gpu.ipynb)

# Contextualized Topic Models, CTM 설치

contextualized topic model 라이브러리를 설치합시다.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install contextualized-topic-models==2.2.0

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pyldavis

# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

"""## 노트북 재시작

원활한 실습을 위해서 노트북을 재시작 할 필요가 있습니다.

상단에서 런타임 > 런타임 재시작을 클릭해주세요.

# 데이터

학습을 위한 데이터가 필요합니다. 여기서는 하나의 라인(line)에 하나의 문서로 구성된 파일이 필요한데요. 우선, 여러분들의 데이터가 없다면 여기서 준비한 파일로 실습을 해봅시다.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !wget https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt

!head -n 1 2016-10-20.txt

!head -n 3 2016-10-20.txt

!head -n 5 2016-10-20.txt

!head -n 20 2016-10-20.txt

text_file = "2016-10-20.txt"

"""# 필요한 것들을 임포트"""

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, bert_embeddings_from_list
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Mecab
from tqdm import tqdm

"""## 전처리"""

documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]

documents[:5]

not '19  1990  52 1 22'.replace(' ', '').isdecimal()

preprocessed_documents = []

for line in tqdm(documents):
  # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
  if line and not line.replace(' ', '').isdecimal():
    preprocessed_documents.append(line)

len(preprocessed_documents)

class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result

custom_tokenizer = CustomTokenizer(Mecab())

vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

train_bow_embeddings = vectorizer.fit_transform(preprocessed_documents)

print(train_bow_embeddings.shape)

vocab = vectorizer.get_feature_names()
id2token = {k: v for k, v in zip(range(0, len(vocab)), vocab)}

len(vocab)

train_contextualized_embeddings = bert_embeddings_from_list(preprocessed_documents, \
                                                            "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

qt = TopicModelDataPreparation()

training_dataset = qt.load(train_contextualized_embeddings, train_bow_embeddings, id2token)

"""## Combined TM 학습하기
이제 토픽 모델을 학습합니다. 여기서는 하이퍼파라미터에 해당하는 토픽의 개수(n_components)로는 50개를 선정합니다.
"""

ctm = CombinedTM(bow_size=len(vocab), contextual_size=768, n_components=50, num_epochs=20)
ctm.fit(training_dataset)

"""# 토픽들

학습 후에는 토픽 모델이 선정한 토픽들을 보려면 아래의 메소드를 사용합니다.

```
get_topic_lists
```
해당 메소드에는 각 토픽마다 몇 개의 단어를 보고 싶은지에 해당하는 파라미터를 넣어즐 수 있습니다.
"""

ctm.get_topics(5)

ctm.get_topics(10)

"""# 시각화

우리의 토픽들을 시각화하기 위해서는 PyLDAvis를 사용합니다.

위에서 출력한 토픽 번호는 pyLDAvis에서 할당한 토픽 번호와 일치하지 않으므로 주의합시다.  
가령, 48번 토픽이었던 ['원유', '유가', '뉴욕', '오른', '연방', '마쳤', '서부', '달러', '51', '지수']가 아래의 PyLDAvis에서는 24번 토픽이 되었습니다.
"""

import pyLDAvis as vis

lda_vis_data = ctm.get_ldavis_data_format(vocab, training_dataset, n_samples=10)

ctm_pd = vis.prepare(**lda_vis_data)
vis.display(ctm_pd)

"""참고 자료 : https://github.com/MilaNLProc/contextualized-topic-models"""