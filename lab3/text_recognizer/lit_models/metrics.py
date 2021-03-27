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
            distance = editdistance.distance(pred, target) # 한 문자열을 다른 문자열로 바꾸는데 필요한 add edit delete의 수 
            error = distance / max(len(pred), len(target))
            self.error = self.error + error # 0으로 initialize되어있나?
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total


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
