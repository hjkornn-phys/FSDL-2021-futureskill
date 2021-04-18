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
        # 공백이 생기는 인덱스 저장
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
            #   "sadsadfasasf asd gg" 에서 max length를 4로 주면 (start_ind, end_ind)가 (0, 13)으로 결정되었을때 에러. 
            try:
                ind = np.random.randint(0, len(self.word_start_inds) - 1) # 시작하는 인덱스 지정, 마지막 공백은 제외
                start_ind = self.word_start_inds[ind]
                end_ind_candidates = []
                for ind in range(ind + 1, len(self.word_start_inds)): # 끝나는 인덱스 지정, 시작 바로 다음 공백부터 마지막 공백 포함
                    if self.word_start_inds[ind] - start_ind > max_length + 1:  # +1이 들어가야 하지 않나???- 수정완료
                        break
                    end_ind_candidates.append(self.word_start_inds[ind]) 
                end_ind = np.random.choice(end_ind_candidates)
                sampled_text = self.text[start_ind:end_ind].strip()
                return sampled_text
            except Exception:  # pylint: disable=broad-except
                pass
        raise RuntimeError("Was not able to generate a valid string")


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
