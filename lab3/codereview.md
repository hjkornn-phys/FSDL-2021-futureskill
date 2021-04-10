# text_recognizer/lit_models/metrics.py
```python
from typing import Sequence

import pytorch_lightning as pl
import torch
import editdistance


class CharacterErrorRate(pl.metrics.Metric):
    """Character error rate metric, computed using Levenshtein distance."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for ind in range(N):
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens] # x = torch([0,2,1]) -> [2]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            distance = editdistance.distance(pred, target) 
            error = distance / max(len(pred), len(target))
            self.error = self.error + error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total
```
pytorch_lightning.metrics.Metric를 상속받는 class CharacterErrorRate를 정의합니다. 이 클래스는 multi-gpu에서도 metric이 동기화되어 작동하도록 합니다.
add_state method를 통해 매 단계 update할 변수와 기본값, 그리고 분배된 metric의 결과를 어떤 방식으로 reduce(합치는 연산) 할지를 인자로 받습니다.

blank, start, end, padding을 무시하고 target과 pred의 차이를 editdistance(한 sequence에서 다른 sequence로 변환할 때 필요한 add delete edit 의 수) 로 정량화합니다.

```python
def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],  # error will be 0
            [0, 2, 1, 1, 1, 1],  # error will be .75
            [0, 2, 2, 4, 4, 1],  # error will be .5
        ]
    )
    Y = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
        ]
    )
    metric(X, Y)
    print(metric.compute())
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3


if __name__ == "__main__":
    test_character_error_rate()
```

# text_recognizer/lit_models/util.py
```python
from typing import Union
import torch


def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    Return indices of first occurence of element in x. If not found, return length of x along dim.

    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9

    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element # torch.tensor([[2,3,4,3]])== 3 -> tensor([[False, True, False, True]])  
    #  tensor([[False, True, False, True]]).cumsum(dim) = [[0, 1, 1, 2]]
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices # 각 항에서 element가 처음 나오는 index 추출, 없으면 0
    ind[ind == 0] = x.shape[dim]
    return ind
```
dim dimension 기준으로 (default==1) element가 가장 먼저 나오는 index를 반환합니다. 만약 element가 존재하지 않으면 dim dimension의 길이를 반환합니다.

# text_recognizer/lit_models/ctc.py
```python
import argparse
import itertools
import torch

from .base import BaseLitModel
from .metrics import CharacterErrorRate
from .util import first_element


def compute_input_lengths(padded_sequences: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    padded_sequences
        (N, S) tensor where elements that equal 0 correspond to padding

    Returns
    -------
    torch.Tensor
        (N,) tensor where each element corresponds to the non-padded length of each sequence

    Examples
    --------
    >>> X = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0], [1, 2, 3, 0, 5]])
    >>> compute_input_lengths(X)
    tensor([2, 3, 5])
    """
    lengths = torch.arange(padded_sequences.shape[1]).type_as(padded_sequences)
    return ((padded_sequences > 0) * lengths).argmax(1) + 1
```
`compute_input_lengths` torch.tensor의 1 dimension에 nonzero masking하고 (`[2, 3, 5, 4, 0 ,2] -> [True, True, True, True, False, True]`) , `length = [0, 1, 2, 3, 4, 5]` 와 곱해 
`[0, 1, 2, 3, 0, 5]`에서 가장 큰 값을 갖는 index + 1 을 반환합니다. 따라서 padding이 0이라고 할 때, (padding이지 않은 가장 마지막 요소의 index + 1)을 반환합니다. 


```python
class CTCLitModel(BaseLitModel):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        inverse_mapping = {val: ind for ind, val in enumerate(self.model.data_config["mapping"])} 
        start_index = inverse_mapping["<S>"]
        self.blank_index = inverse_mapping["<B>"]
        end_index = inverse_mapping["<E>"]
        self.padding_index = inverse_mapping["<P>"]

        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)
        # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

        ignore_tokens = [start_index, end_index, self.padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)
```

CTCLoss를 loss function으로 갖고 BaseLitModel을 상속받는 CTCLitModel을 정의합니다. self.val_cer 과 self.test_cer는 평균적인 문자 오차율을 보여줍니다.
```python
    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)

        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S  #type_as does not do anything
        target_lengths = first_element(y, self.padding_index).type_as(y)
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        B, _C, S = logprobs.shape

        logprobs_for_loss = logprobs.permute(2, 0, 1)  # -> (S, B, C)
        input_lengths = torch.ones(B).type_as(logprobs_for_loss).int() * S  # All are max sequence length
        target_lengths = first_element(y, self.padding_index).type_as(y)  # Length is up to first padding token
        loss = self.loss_fn(logprobs_for_loss, y, input_lengths, target_lengths)
        self.log("val_loss", loss, prog_bar=True)

        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.val_acc(decoded, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(decoded, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        decoded = self.greedy_decode(logprobs, max_length=y.shape[1])
        self.test_acc(decoded, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(decoded, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

def greedy_decode(self, logprobs: torch.Tensor, max_length: int) -> torch.Tensor: 
        """
        Greedily decode sequences, collapsing repeated tokens, and removing the CTC blank token.

        See the "Inference" sections of https://distill.pub/2017/ctc/

        Using groupby inspired by https://github.com/nanoporetech/fast-ctc-decode/blob/master/tests/benchmark.py#L8

        Parameters
        ----------
        logprobs
            (B, C, S) log probabilities
        max_length
            max length of a sequence

        Returns
        -------
        torch.Tensor
            (B, S) class indices
        """
        B = logprobs.shape[0]
        argmax = logprobs.argmax(1)
        decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.padding_index
        for i in range(B):
            seq = [b for b, _g in itertools.groupby(argmax[i].tolist()) if b != self.blank_index][:max_length]
            for ii, char in enumerate(seq):
                decoded[i, ii] = char
        return decoded
```
model을 통과한 후 log_softmax를 취하면 shape가 (B, C, S) 인 tensor을 얻습니다. C는 num_classes, S는 ctc모델이 예측한 sequence_length입니다.logprobs_for_loss.shape = (S, B, C)인데 b째 sample의 s번째 토큰을 label c라고 예측할 확률의 로그값이 `logprobs_for_loss[s][b][c]`입니다.

`greedy_decode`는 logprobs (logprobs.shape == (B ,C, S))를 입력으로 받습니다. 첫째로 가장 예측 확률이 높은 label을 .argmax(1)로 얻어낸 뒤, (B, max_length)의 모양을 갖고 각 요소는 padding_index인 decoded: torch.tensor를 생성합니다. 

