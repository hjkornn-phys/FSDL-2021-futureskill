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
