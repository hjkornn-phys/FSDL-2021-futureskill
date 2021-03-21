# text_recognizer/data
데이터를 다루기 위한 모든 파일들이 이 디렉토리에 있습니다.

[base_data_module.py](#base_data_module.py)

[util.py](#util.py)

# text_recognizer/lit_models
라이트닝 모듈의 기저 클래스를 정의합니다.

[base.py](#base.py)



# base_data_module.py  
```python
"""Base DataModule class."""
from pathlib import Path
from typing import Dict
import argparse
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from text_recognizer import util


def load_and_print_info(data_module_class: type) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)

def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename


```
필요한 라이브러리들을 import하고 load_and_print_info()를 정의합니다. 이 함수는 파일 하단에 
`if __name__=='__main__'` 구문으로 외부에서 파일 실행 시 자동으로 호출되는 경우도 있습니다 (mnist.py)

add_to_argparse()는 staticmethod로 클래스 내부에서 정의되지만 self를 argument로 받지 않아 클래스 객체 instantiation 없이 호출할 수 있습니다.

_ download_raw_dataset()함수를 정의합니다. 앞의 언더바는 이 함수가 내부 사용을 권장한다는 의미입니다. 지정된 위치에 디렉토리를 만들고, 다운로드 후 처리 과정을 거칩니다.
디렉토리를 만들 때, 부모 디렉토리가 없으면 그것도 생성하고, 이미 있는 디렉토리라면 오류 없이 다음으로 넘어갑니다.



```python
BATCH_SIZE = 128
NUM_WORKERS = 0


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {} # To Dictionary
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        # Make sure to set the variables below in subclasses
        self.dims = None
        self.output_dims = None
        self.mapping = None
```
pytorch_lightning.LightningDataModule을 상속하는 BaseDataModule 클래스를 정의합니다. 이후 사용하는 데이터셋은 모두 BaseDataModule을 상속하므로, 전부 LightningDataModule의
subclass라고 할 수 있습니다.

LightningDataModule을 사용할 때의 장점은 일관된 데이터 준비, 로딩입니다. 모든 데이터셋에 같은 command line parameter를 사용하여 batch size와 num worker들을 정할 수
있고 data split 역시 pytorch_lightning.trainer에 LightningDataModule 객체만 넣어주면 내부에서 setup함수를 통해 정의된 train_dataloader등을 알아서 사용하므로, 코드가 매우 간편해집니다.

emnist.py에서도 setup을 통해 dataloader함수들을 호출하는데, 학습을 담당하는 run_experiment.py에는 명시적으로 setup이나 train_dataloader를 호출하는 코드는 보이지 않습니다. 이는 trainer 내부에서 그 작업이 실행되기 떄문입니다

* vars는 argument를 dictionary로 만들어줍니다.

```python
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
```
data_dirname은 데이터가 존재하는 디렉토리를 지정합니다
함수들 prepare_data, setup 은 상속받는 클래스에서 override해야 합니다.


# util.py
```python
"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch


SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
```

data와 target에 적절한 transform을 가한 뒤 반환합니다. 반드시 `__len__` 함수와 `__getitem`가 반드시 정의되어야 합니다.
```python

def convert_strings_to_labels(strings: Sequence[str], mapping: Dict[str, int], length: int) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels

"""
convert_strings_to_labels("DGOG", {"D":0, "O":5, "G":62 ,"<P>":2, "<S>":3, "<E>":4}, 7)
tensor([[ 3,  0,  4,  2,  2,  2,  2],
        [ 3, 62,  4,  2,  2,  2,  2],
        [ 3,  5,  4,  2,  2,  2,  2],
        [ 3, 62,  4,  2,  2,  2,  2]])
"""
```
의미있는 부분의 앞뒤에 <S>,<E> 토큰을 넣고, 나머지 빈칸에는 <P>(Padding)토큰을 넣어 labeling한 label:list 를 반환합니다. emnist로 문장을 만들 때 필요합니다.
    
    
# base.py
