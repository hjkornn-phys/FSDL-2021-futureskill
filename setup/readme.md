# Setup

딥러닝은 많은 연산량 때문에 GPU, TPU가 필요합니다. 만일 그러한 기기를 이미 가지고 계시다면, [로컬로 사용](#Local)하실 수 있습니다. GPU가 없으시다면, [Google Colab](#Colab) 등을
사용하여 프로젝트를 진행하실 수 있습니다. 

# Colab
구글 계정만 있다면, colab에서 GPU를 무료로 사용할 수 있습니다.
https://colab.research.google.com 에 접속하셔서 '새 노트' 를 클릭하고, '런타임'-'런타임 유형 변경'에서 None을 GPU로 바꿔주세요.
![image](https://user-images.githubusercontent.com/59644774/111866308-e9d0f600-89af-11eb-9c22-ccde4dbcae8f.png)
![image](https://user-images.githubusercontent.com/59644774/111866366-54823180-89b0-11eb-8c50-3628bfc7d2b2.png)

다음으로, `!nvidia-smi`를 입력하고 실행(shift + enter)하시면 연결된 GPU 정보가 확인됩니다.
```
!nvidia-smi
```

![image](https://user-images.githubusercontent.com/59644774/111866526-75975200-89b1-11eb-91ba-fab954e603d4.png)


이제 저장소에서 복제할 차례입니다. 원본 저장소에서는 몇 가지 작은 에러가 있어서 제가 에러픽스한 저장소에서 복제하는 것을 추천드립니다. 
`제 저장소:`
```
# FSDL Spring 2021 Setup
!git clone https://github.com/hjkornn-phys/FSDL-2021-futureskill
%cd FSDL-2021-futureskill
!pip3 install boltons wandb pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
%env PYTHONPATH=.:$PYTHONPATH
```
셀 실행 후 다음 셀에서 pytorch_lightning를 설치합니다. 한 셀에서 설치 시 호환이 불가하다며 에러가 발생합니다.
```
!pip3 install pytorch_lightning==1.1.4
```

`원본 저장소:`
```
# FSDL Spring 2021 Setup
!git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
%cd fsdl-text-recognizer-2021-labs
!pip3 install boltons wandb pytorch_lightning==1.1.4 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchtext==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
%env PYTHONPATH=.:$PYTHONPATH
```   

## mnist.py 실행
좌측의 폴더 모양 아이콘을 클릭하면 다운로드 된 파일들을 볼 수 있습니다.
그러면 lab1에 접근하여 mnist.py를 실행해 볼까요?

`
%cd lab1/text_recognizer/data/
`

`
run mnist.py
`
![image](https://user-images.githubusercontent.com/59644774/111867946-a0d26f00-89ba-11eb-9b2b-5b2719927e03.png)


에러가 발생하는 이유는 mnist를 다운받는 링크(LeCun 블로그)가 만료되었기 때문입니다. 본격적으로 text recognization에 필요한 emnist는 잘 다운로드 되므로, 여기까지 오셨다면 셋업이 잘 진행된 것을 확인한 것입니다. 이 노트북을 구글 드라이브에 저장하시면 다시 꺼내서 사용하실 수 있습니다




# Local

 setup 폴더의 [Originalreadme](https://github.com/hjkornn-phys/FSDL-2021-futureskill/blob/main/setup/Originalreadme.md)
의 local부분을 참고하여 셋업하실 수 있습니다. 
