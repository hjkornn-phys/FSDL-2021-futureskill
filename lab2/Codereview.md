# text_recognizer/data/sentence_generator.py

```python
"""SentenceGenerator class and supporting functions."""
import itertools
import re
import string
from typing import Optional

import nltk
import numpy as np

from text_recognizer.data import BaseDataModule

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "nltk"


class SentenceGenerator:
    """Generate text sentences using the Brown corpus."""

    def __init__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_start_inds = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)]
        # 공백이 생기는 인덱스+1 == 단어 시작 인덱스 리스트로 저장
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """
        Sample a string from text of the Brown corpus of length at least one word and at most max_length.
        """
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("Must provide max_length to this method or when making this object.")
        # if 문 연달아 두 개를 써서 self.max_length 를 max_length에 할당. 그랬는데도 없으면 에러

        for ind in range(10):  # Try several times to generate before actually erroring
            #   "beautiful soup challenge" 에서 max length를 4로 주면 (start_ind, end_ind)가 (0, 10)으로 결정되었을때 에러. 
            try:
                ind = np.random.randint(0, len(self.word_start_inds) - 1) # 시작하는 인덱스 지정
                start_ind = self.word_start_inds[ind]
                end_ind_candidates = []
                for ind in range(ind + 1, len(self.word_start_inds)): # 끝나는 인덱스 지정
                    if self.word_start_inds[ind] - start_ind > max_length+1:  # +1은 공백이 포함되므로
                        break
                    end_ind_candidates.append(self.word_start_inds[ind]) 
                end_ind = np.random.choice(end_ind_candidates)
                sampled_text = self.text[start_ind:end_ind].strip()
                return sampled_text
            except Exception:  # pylint: disable=broad-except
                pass
        raise RuntimeError("Was not able to generate a valid string")
```
`re.finditer`정규표현식 모듈을 사용하여 공백에 해당하는 모든 문자열을 iterator로 반환합니다. 각 요소는 match 객체인데, start attribute를 사용하여 어디에서 공백이 발생했는지 찾고, +1으로 단어가 
시작하는 index를 모읍니다.

`generate` if문 두 개를 연달아 사용하여, SentenceGenerator를 instantiate 할 때, max_length를 지정하지 않은 경우, generate에서 얻은 인자만으로 지정할 수 있게 했습니다.

`generate`내에 word_start_index는 문자열에서 단어가 시작하는 인덱스를 list로 저장하고, 그 안에서 sampling을 시작할 지점을 start_ind로 추출한 뒤, sample의 길이가 max_length를
넘지 않게 하는 end_ind_candidates를 얻고, sampling을 끝낼 지점을 추출합니다. max_length보다 긴 문자열이 sampling되는 경우도 존재하므로, 10번의 sampling 을 시켜 10번 연속으로 실패하지 않는다면 에러 없이
프로그램이 실행되도록 하였습니다

```python
def brown_text():
    """Return a single string with the Brown corpus with all punctuation stripped."""
    sents = load_nltk_brown_corpus()
    text = " ".join(itertools.chain.from_iterable(sents))
    text = text.translate({ord(c): None for c in string.punctuation})
    text = re.sub("  +", " ", text)
    return text


def load_nltk_brown_corpus():
    """Load the Brown corpus using the NLTK library."""
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()

if __name__ =="__main__":
    SentenceGenerator(15)
```
`brown_text` itertools.chain.from_iterable(): 여러 iterable에서 요소를 lazy evaluation(generator) 방식으로 하나씩 반환합니다. str.translate는 딕셔너리{a:b}를 받아 a를 b로 변환하는 
메소드입니다. 이때 key와  value 모두 유니코드 

