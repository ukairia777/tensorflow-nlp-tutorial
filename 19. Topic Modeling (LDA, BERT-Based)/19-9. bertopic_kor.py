# -*- coding: utf-8 -*-
"""Bertopic_Korean.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ll_J91K0v76OtiwNu5jGBg92MfEbTUZ_
"""

!pip install bertopic

!pip install bertopic[visualization]

# Commented out IPython magic to ensure Python compatibility.
# Colab에 Mecab 설치
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# %cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

"""# 데이터

학습을 위한 데이터가 필요합니다. 여기서는 하나의 라인(line)에 하나의 문서로 구성된 파일이 필요한데요. 우선, 여러분들의 데이터가 없다면 여기서 준비한 파일로 실습을 해봅시다.
"""

!wget https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt

!head -n 5 2016-10-20.txt

text_file = "2016-10-20.txt"

from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

docs[:5]

"""# 필요한 것들을 임포트"""

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Mecab
from bertopic import BERTopic

"""# 전처리"""

documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]

preprocessed_documents = []

for line in tqdm(documents):
  # 빈 문자열이거나 숫자로만 이루어진 줄은 제외
  if line and not line.replace(' ', '').isdecimal():
    preprocessed_documents.append(line)

preprocessed_documents[:5]

"""# Mecab과 SBERT를 이용한 Bertopic"""

class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result

custom_tokenizer = CustomTokenizer(Mecab())

vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", \
                 vectorizer_model=vectorizer,
                 nr_topics=50,
                 top_n_words=10,
                 calculate_probabilities=True)

topics, probs = model.fit_transform(preprocessed_documents)

model.visualize_topics()

model.visualize_distribution(probs[0])

for i in range(0, 50):
  print(i,'번째 토픽 :', model.get_topic(i))