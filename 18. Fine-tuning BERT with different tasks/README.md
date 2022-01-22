# Tensorflow-NLP-tutorial
다음의 안내를 읽어보고 실습을 진행해보세요.

# 1. model_from_transformers  

transformers 라이브러리에서는 각종 태스크에 맞게 BERT 위에 출력층을 추가한 모델 클래스 구현체를 제공하고 있습니다. 아래의 구현체를 사용하면 사용자가 별도의 출력층을 설계할 필요없이 태스크에 맞게 모델을 로드하여 사용할 수 있습니다. 저는 모델 구조의 이해를 위해 모든 실습에서 출력층을 직접 설계한 버전의 코드를 작성하였지만, 이미 모델의 구조를 이해한 상황에서는 아래와 같이 이미 출력층이 설계된 모델들을 사용하는 것이 훨씬 코드 작성이 간편합니다. 깃허브에서 파일명에 'model_from_transformers'가 들어간 파일들이 아래의 구현체들을 사용한 버전의 실습 파일입니다.  

```
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("모델 이름", num_labels=분류할 레이블의 개수)
```
```
from transformers import TFBertForTokenClassification

model = TFBertForTokenClassification.from_pretrained("모델 이름", num_labels=분류할 레이블의 개수)
```
```
from transformers import TFBertForQuestionAnswering

model = TFBertForQuestionAnswering.from_pretrained('모델 이름')
```



# 2. Colab에서의 TPU 사용  

18챕터의 모든 BERT 실습은 Colab에서 진행한다고 가정합니다.  

* 파일명이 tpu로 끝나는 경우 Colab > 런타임 > 런타임 유형 변경에서 'TPU'를 선택 후 실습을 진행해주세요.
* 파일명이 gpu로 끝나는 경우 Colab > 런타임 > 런타임 유형 변경에서 'GPU'를 선택 후 실습을 진행해주세요.

모든 TPU 코드는 아래의 링크를 참고하여 작성되었습니다.  
아래 링크에서 안내하는 TPU 코드만 제거하면 TPU 실습 코드를 GPU 실습 코드로 변경할 수 있습니다.  
다시 말해 파일명이 tpu로 끝나는 파일에서 아래의 링크에서 설명하는 코드들을 전부 제거하면 GPU에서 실습해도 됩니다.  

링크 : https://wikidocs.net/119990
