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
메소드입니다. 이때 key와  value 모두 유니코드 값입니다.

`re.sub()` 첫번째 인자를 두 번째 인자로 교체하는 함수입니다.

# text_recognizer/data/emnist.py
```python
"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
from pathlib import Path
from typing import Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import numpy as np
import toml
import torch

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist" # 데이터를 저장할 위치
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml" 
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5" # 처리 후 파일명
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json" # 다운로드 및 처리 후 생성되는 파일의 경로와 이름


class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)} # {"<B>": 0, ...}
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])  # Extraction
        # Extra dimension is added by ToTensor()
        self.output_dims = (1,)

    def prepare_data(self): # init과 겹침
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f) 

    def setup(self, stage: str = None): # None 이면 둘 다 하는거구나
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            train_size = int(TRAIN_FRAC * len(data_trainval))
            val_size = len(data_trainval) - train_size
            self.data_train, self.data_val = torch.utils.data.random_split(
                data_trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic # load하기 전

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data # load후 포함
```
`EMNIST` ESSENTIAL_FILENAME을 가진 파일이 있는지 확인 후, 없다면 다운로드 후 처리하여 ESSENTIAL_FILENAME을 가진 파일을 생성하고, 있으면 그 정보를 바탕으로 데이터모듈을 구축합니다.

mapping은 분류할 클래스들의 리스트이고, inverse_mapping은 딕셔너리 형태로 {0: 'A', 1:'B',...} index에 mapping의 요소를 대응합니다.

```python
def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping EMNIST...")
    curdir = os.getcwd()
    os.chdir(dirname)
    zip_file = zipfile.ZipFile(filename, "r")
    zip_file.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat  # pylint: disable=import-outside-toplevel

    # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS # 오른쪽으로 4칸씩 shift
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True) # make parent dir if needed, do not make dir if it already exists
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("Saving essential dataset parameters to text_recognizer/datasets...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]} # dataset mapping
    characters = _augment_emnist_characters(mapping.values())
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])} # batch size not needed
    with open(ESSENTIALS_FILENAME, "w") as f: 
        json.dump(essentials, f) #wirte essentials to ESSENTIALS_FILENAME

    print("Cleaning up...")
    shutil.rmtree("matlab") # Remove downloaded data
    os.chdir(curdir) # 원래 위치로
```
`_process_raw_dataset` 지정한 디렉토리로 이동한 뒤 압축파일에서 "matlab/emnist-byclass.mat"을 압축해제합니다. 그 후 scipy.io.loadmat으로 데이터를 열고, 적절한 차원으로 reshape합니다. ctc loss를 계산할 때 쓰이는 SPECIAL_TOKENS가 4개이고, 각각 inverse_mapping에서 가장 먼저 등장하므로,target의 모든 값들에 대해 +4의 right shift가 필요합니다.

데이터가 unbalanced이므로  `_sample_to_balance` 를 사용하여 균형을 맞추어줍니다. characters 에 iam dataset에서 쓰이는 분류 클래스(따옴표 등)와 ctc 토큰을 추가해줍니다.
characters와 input_shape를 ESSENTIALS_FILENAME으로 저장합니다. 압축을 풀면서 생성된 matlab 폴더를 삭제하고, 원래 있던 디렉토리로 이동합니다
```python
def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean()) # 각 라벨당 몇 개 씩의 X가 존재하는지 세고, 그 평균(=라벨당 X의 수)을 정수로 반환
    all_sampled_inds = []
    for label in np.unique(y.flatten()): # y에 존재하는 라벨 마다..
        inds = np.where(y == label)[0] # return nonzero indicies that are labeled as label
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample)) # 
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augment the mapping with extra symbols."""
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Also add special tokens:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


if __name__ == "__main__":
    load_and_print_info(EMNIST)
```
`_sample_to_balance`num_to_sample에 label당 개수의 평균에 가까운 정수를 저장합니다. y에 존재하는 label마다 label의 값을 갖는 index를 inds에 저장하고, inds 내에서 num_to_sample(평균)번 복원추출하여 어떤 index를 사용할지 정합니다.(`y = torch.tensor([[1, 1, 2, 1, 2, 3]])`일때, num_to_sample==2이고, label==1 일 때 inds = `[0, 1, 3]`이므로 inds에서 2번 복원 추출한 결과, 즉 `[0,0] or [1, 3] ...` 에 np.unique()를 적용한 것을 사용합니다) Dataset 내에서 정해진 index들만을 사용합니다.

# text_recognizer/data/emnist_lines.py

```python
from typing import Dict, Sequence
from collections import defaultdict
from pathlib import Path
import argparse

from torchvision import transforms
import h5py
import numpy as np
import torch

from text_recognizer.data.util import BaseDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST


DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_lines_essentials.json"

MAX_LENGTH = 32
MIN_OVERLAP = 0
MAX_OVERLAP = 0.33
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000


class EMNISTLines(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""

    def __init__(
        self,
        args: argparse.Namespace = None,
    ):
        super().__init__(args)

        self.max_length = self.args.get("max_length", MAX_LENGTH)
        self.min_overlap = self.args.get("min_overlap", MIN_OVERLAP)
        self.max_overlap = self.args.get("max_overlap", MAX_OVERLAP)
        self.num_train = self.args.get("num_train", NUM_TRAIN)
        self.num_val = self.args.get("num_val", NUM_VAL)
        self.num_test = self.args.get("num_test", NUM_TEST)
        self.with_start_end_tokens = self.args.get("with_start_end_tokens", False)

        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping
        self.dims = (
            self.emnist.dims[0],
            self.emnist.dims[1],
            self.emnist.dims[2] * self.max_length,
        )
        self.output_dims = (self.max_length, 1)
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.transform = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max line length in characters.")
        parser.add_argument(
            "--min_overlap",
            type=float,
            default=MIN_OVERLAP,
            help="Min overlap between characters in a line, between 0 and 1.",
        )
        parser.add_argument(
            "--max_overlap",
            type=float,
            default=MAX_OVERLAP,
            help="Max overlap between characters in a line, between 0 and 1.",
        )
        parser.add_argument("--with_start_end_tokens", action="store_true", default=False)
        return parser

    @property
    def data_filename(self):
        return (
            DATA_DIRNAME
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}.h5"
        )

    def prepare_data(self) -> None:
        if self.data_filename.exists():
            return
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")

    def setup(self, stage: str = None) -> None:
        print("EMNISTLinesDataset loading data from HDF5...")
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = f["y_train"][:].astype(int)
                x_val = f["x_val"][:]
                y_val = f["y_val"][:].astype(int)

            self.data_train = BaseDataset(x_train, y_train, transform=self.transform)
            self.data_val = BaseDataset(x_val, y_val, transform=self.transform)

        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = f["y_test"][:].astype(int)
            self.data_test = BaseDataset(x_test, y_test, transform=self.transform)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "EMNIST Lines Dataset\n"  # pylint: disable=no-member
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data
```
BaseDataModule을 상속받는 EMNISTLines class를 정의합니다. EMNIST와 유사하지만 command line parameter로 --max_length, --min_overlap, --max_overlap을 받고, output dimension은 (max_length,1)이 됩니다.


    def _generate_data(self, split: str) -> None:
        print(f"EMNISTLinesDataset generating data for {split}...")

        # pylint: disable=import-outside-toplevel
        from text_recognizer.data.sentence_generator import SentenceGenerator

        sentence_generator = SentenceGenerator(self.max_length - 2)  # Subtract two because we will add start/end tokens

        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()

        if split == "train":
            samples_by_char = get_samples_by_char(emnist.x_trainval, emnist.y_trainval, emnist.mapping)
            num = self.num_train
        elif split == "val":
            samples_by_char = get_samples_by_char(emnist.x_trainval, emnist.y_trainval, emnist.mapping)
            num = self.num_val
        else:
            samples_by_char = get_samples_by_char(emnist.x_test, emnist.y_test, emnist.mapping)
            num = self.num_test

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            x, y = create_dataset_of_images(
                num, samples_by_char, sentence_generator, self.min_overlap, self.max_overlap, self.dims
            )
            y = convert_strings_to_labels( # "The Cat" -> 
                y,
                emnist.inverse_mapping,
                length=self.output_dims[0],
                with_start_end_tokens=self.with_start_end_tokens,
            )
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}", data=y, dtype="u1", compression="lzf")
```python

def get_samples_by_char(samples, labels, mapping):
    samples_by_char = defaultdict(list) 
    for sample, label in zip(samples, labels): 
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char # {'A':[A_1, A_2], 'B':[B_1], 'C':[C_1]} (전체에 대해서)


def select_letter_samples_for_string(string, samples_by_char): # 입력[1]은 get_samples_by_char()
    zero_image = torch.zeros((28, 28), dtype=torch.uint8)
    sample_image_by_char = {}
    for char in string:
        if char in sample_image_by_char:
            continue
        samples = samples_by_char[char]
        sample = samples[np.random.choice(len(samples))] if samples else zero_image # 수많은 A중 하나 고르기
        sample_image_by_char[char] = sample.reshape(28, 28) # character 하나 당 이미지 하나 
    return [sample_image_by_char[char] for char in string]


def construct_image_from_string(
    string: str, samples_by_char: dict, min_overlap: float, max_overlap: float, width: int
) -> torch.Tensor:
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = select_letter_samples_for_string(string, samples_by_char)
    N = len(sampled_images)
    H, W = sampled_images[0].shape
    next_overlap_width = W - int(overlap * W)
    #dataset(n, H, W)에서 shape(-1)로 입력받음
    concatenated_image = torch.zeros((H, width), dtype=torch.uint8)
    x = 0
    for image in sampled_images:
        concatenated_image[:, x : (x + W)] += image
        x += next_overlap_width
    return torch.minimum(torch.Tensor([255]), concatenated_image) 


def create_dataset_of_images(N, samples_by_char, sentence_generator, min_overlap, max_overlap, dims):
    images = torch.zeros((N, dims[1], dims[2]))
    labels = []
    for n in range(N):
        label = sentence_generator.generate()
        images[n] = construct_image_from_string(label, samples_by_char, min_overlap, max_overlap, dims[-1])
        labels.append(label)
    return images, labels


def convert_strings_to_labels(
    strings: Sequence[str], mapping: Dict[str, int], length: int, with_start_end_tokens: bool
) -> np.ndarray:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = np.ones((len(strings), length), dtype=np.uint8) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        if with_start_end_tokens:
            tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels


if __name__ == "__main__":
    load_and_print_info(EMNISTLines)
```
`get_samples_by_char` 빈 리스트 `[]`을 기본값으로 하는 DefaultDict 객체 samples_by_char를 생성합니다. 이 객체에 존재하지 않는 key를 통해 호출하면, 해당 key의 value는 `[]`가 됩니다. samples_by_char의 key에 char('A', 'B',...)를 넣고, 그 value는 sample들을 요소로 같는 list가 됩니다. 

`select_letter_samples_for_string` string에 존재하는 각 character마다 `samples_by_char[char]`로 char에 해당하는 sample의 list, samples를 얻습니다. samples에서 하나를 뽑아 28x28 형태로 reshape하고 `sample_image_by_char`에 넣습니다. 이때 sample_by_char은 dict type으로 {char: sample} 로 저장됩니다. 만약 string 내에 같은 글자가 있었다면, ex) 'peace' 에서 두번째 문자 e와 다섯번째 글자 e는 같은 image로 구현됩니다. 

`select_letter_samples_for_string`은 string에 있는 char마다 해당하는 sample image의 list를 반환합니다. 'cat' -> `[sample_image_for_c: torch.tensor, sample_image_for_a: torch.tensor, sample_image_for_t: torch.tensor]`

`construct_image_from_string` width만큼의 가로를 갖는 concatenated_image를 0으로 초기화하고, overlap의 비율만큼 겹치는 image를 생성합니다. 255를 넘는 값이 있다면 그 값을 255로 변화시킵니다. 두 이미지의 겹치는 부분에서 이러한 일이 발생할 수 있습니다.

`create_dataset_of_images` N개의 문장과 그에 해당하는 image를 만들어냅니다.

`convert_strings_to_labels` padding에 해당하는 label (`mapping["<P>"]`) 로 labels를 초기화하고 string의 모든 요소를 원소가 하나인 list로 만듭니다. (('what', 'up') ->`['w','h','a','t'], ['u','p']`) 그 후 start token과 end token을 추가합니다 (`['<S>', 'w','h','a','t', '<E>'], ['<S>', 'u', 'p', '<E>']`)
`labels[i, ii]`를 `mapping[token]`으로 대체합니다. 따라서 `labels`는 `[['<S>', 'w','h','a','t', '<E>', '<P>', '<P>'], ['<S>', 'u', 'p', '<E>','<P>','<P>','<P>']
]`의 각 요소를 mapping에 의해 해당하는 정수로 변환한 리스트가 됩니다.

