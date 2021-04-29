# text_recognizer/data/iam_original_and__synthetic_paragraphs.py
```python
"""IAM Original and Synthetic Paragraphs Dataset class."""
import argparse
from torch.utils.data import ConcatDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs


class IAMOriginalAndSyntheticParagraphs(BaseDataModule):
    """A concatenation of original and synthetic IAM paragraph datasets."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)

        self.iam_paragraphs = IAMParagraphs(args)
        self.iam_syn_paragraphs = IAMSyntheticParagraphs(args)

        self.dims = self.iam_paragraphs.dims
        self.output_dims = self.iam_paragraphs.output_dims
        self.mapping = self.iam_paragraphs.mapping
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        self.iam_paragraphs.prepare_data()
        self.iam_syn_paragraphs.prepare_data()

    def setup(self, stage: str = None) -> None:
        self.iam_paragraphs.setup(stage)
        self.iam_syn_paragraphs.setup(stage)

        self.iam_paragraphs.setup('test') #에러고침

        self.data_train = ConcatDataset([self.iam_paragraphs.data_train, self.iam_syn_paragraphs.data_train])
        self.data_val = self.iam_paragraphs.data_val
        self.data_test = self.iam_paragraphs.data_test

    # TODO: can pass multiple dataloaders instead of concatenation datasets
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multiple_loaders.html#multiple-training-dataloaders
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
    #     )

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Original and Synthetic Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


if __name__ == "__main__":
    load_and_print_info(IAMOriginalAndSyntheticParagraphs)
```
39번째 줄`self.iam_paragraphs.setup('test')` 없이 이 module이 호출되면 IAMparaghs.data_train이 존재하지 않는다는 에러와 함께 training이 진행되지 않습니다. stage=None으로 초기화되어있으므로 stage=='fit'인
경우와 stage=='test'인 경우가 모두 실행되어야 하지만, 후자는 실행되지 않아서 에러가 발생했고, 명시적으로 stage인자에 'test'를 주었습니다.

# text_recognizer/models/resnet_transformer.py

```python3
import argparse
from typing import Any, Dict
import math
import torch
import torch.nn as nn
import torchvision

from .transformer_util import PositionalEncodingImage, PositionalEncoding, generate_square_subsequent_mask


TF_DIM = 256
TF_FC_DIM = 1024
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4
RESNET_DIM = 512  # hard-coded


class ResnetTransformer(nn.Module):
    """Process the line through a Resnet and process the resulting sequence with a Transformer decoder"""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.input_dims = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])
        inverse_mapping = {val: ind for ind, val in enumerate(data_config["mapping"])}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = data_config["output_dims"][0]
        self.args = vars(args) if args is not None else {}

        self.dim = self.args.get("tf_dim", TF_DIM)
        tf_fc_dim = self.args.get("tf_fc_dim", TF_FC_DIM)
        tf_nhead = self.args.get("tf_nhead", TF_NHEAD)
        tf_dropout = self.args.get("tf_dropout", TF_DROPOUT)
        tf_layers = self.args.get("tf_layers", TF_LAYERS)

        # ## Encoder part - should output  vector sequence of length self.dim per sample
        resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))  # Exclude AvgPool and Linear layers
        # Resnet will output (B, RESNET_DIM, _H, _W) logits where _H = input_H // 32, _W = input_W // 32

        # self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=1)
        # encoder_projection will output (B, dim, _H, _W) logits

        self.enc_pos_encoder = PositionalEncodingImage(
            d_model=self.dim, max_h=self.input_dims[1], max_w=self.input_dims[2]
        )  # Max (Ho, Wo)

        # ## Decoder part
        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)

        self.dec_pos_encoder = PositionalEncoding(d_model=self.dim, max_len=self.max_output_length)

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=tf_nhead, dim_feedforward=tf_fc_dim, dropout=tf_dropout),
            num_layers=tf_layers,
        )

        self.init_weights()  # This is empirically important

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        nn.init.kaiming_normal_(self.encoder_projection.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if self.encoder_projection.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # pylint: disable=protected-access
                self.encoder_projection.weight.data
            )
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_projection.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (Sx, B, E) logits
        """
        _B, C, _H, _W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)  # (B, RESNET_DIM, _H // 32, _W // 32),   (B, 512, 18, 20) in the case of IAMParagraphs
        x = self.encoder_projection(x)  # (B, E, _H // 32, _W // 32),   (B, 256, 18, 20) in the case of IAMParagraphs

        # x = x * math.sqrt(self.dim)  # (B, E, _H // 32, _W // 32)  # This prevented any learning
        x = self.enc_pos_encoder(x)  # (B, E, Ho, Wo);     Ho = _H // 32, Wo = _W // 32
        x = torch.flatten(x, start_dim=2)  # (B, E, Ho * Wo)
        x = x.permute(2, 0, 1)  # (Sx, B, E);    Sx = Ho * Wo
        return x

    def decode(self, x, y):
        """
        Parameters
        ----------
        x
            (B, H, W) image
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (Sy, B, C) logits
        """
        y_padding_mask = y == self.padding_token
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.dec_pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        output = self.transformer_decoder(
            tgt=y, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=y_padding_mask
        )  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (B, C, Sy) logits
        """
        x = self.encode(x)  # (Sx, B, E)
        output = self.decode(x, y)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]
        S = self.max_output_length
        x = self.encode(x)  # (Sx, B, E)

        output_tokens = (torch.ones((B, S)) * self.padding_token).type_as(x).long()  # (B, S)
        output_tokens[:, 0] = self.start_token  # Set start token
        for Sy in range(1, S):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(x, y)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            
            output_tokens[:, Sy : Sy + 1] = output.flatten()[-1:]  # Set the last output token
            # 에러고침

            # Early stopping of prediction loop to speed up prediction
            if ((output_tokens[:, Sy] == self.end_token) | (output_tokens[:, Sy] == self.padding_token)).all():
                break

        # Set all tokens after end token to be padding
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token

        return output_tokens  # (B, Sy)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tf_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_fc_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_dropout", type=float, default=TF_DROPOUT)
        parser.add_argument("--tf_layers", type=int, default=TF_LAYERS)
        parser.add_argument("--tf_nhead", type=int, default=TF_NHEAD)
        return parser

```
LineCNN을 encoder로 사용한 transformer model과 비슷하지만 차이가 있습니다. 이 경우 RESNET을 통과 시킨 뒤 Embeddig 차원에 맞게 projection합니다. `forward`에서 flatten하지 않는다면
차원이 맞지 않아 에러가 나게 됩니다.


# training/save_best_model.py
```python3
"""
Module to find and save the best model trained on a given dataset to artifacts directory.
Run:
python training/save_best_model.py --entity=fsdl-user \
                                   --project=fsdl-text-recognizer-2021-labs \
                                   --trained_data_class=IAMLines
To find entity and project, open any wandb run in web browser and look for the field "Run path" in "Overview" page.
"Run path" is of the format "<entity>/<project>/<run_id>".
"""
import argparse
import sys
import shutil
import json
from pathlib import Path
import tempfile
from typing import Optional, Union
import wandb


FILE_NAME = Path(__file__).resolve()
ARTIFACTS_BASE_DIRNAME = FILE_NAME.parents[1] / "text_recognizer" / "artifacts"
TRAINING_LOGS_DIRNAME = FILE_NAME.parent / "logs"


def save_best_model():
    """Find and save the best model trained on a given dataset to artifacts directory."""
    parser = _setup_parser()
    args = parser.parse_args()

    if args.mode == "min":
        default_metric_value = sys.maxsize
        sort_reverse = False
    else:
        default_metric_value = 0
        sort_reverse = True

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"config.data_class": args.trained_data_class})
    sorted_runs = sorted(
        runs,
        key=lambda run: _get_summary_value(wandb_run=run, key=args.metric, default=default_metric_value),
        reverse=sort_reverse,
    )

    best_run = sorted_runs[0]
    summary = best_run.summary
    print(f"Best run ({best_run.name}, {best_run.id}) picked from {len(runs)} runs with the following metrics:")
    print(f" - val_loss: {summary['val_loss']}, val_cer: {summary['val_cer']}, test_cer: {summary['test_cer']}")

    artifacts_dirname = _get_artifacts_dirname(args.trained_data_class)
    with open(artifacts_dirname / "config.json", "w") as file:
        json.dump(best_run.config, file, indent=4)
    with open(artifacts_dirname / "run_command.txt", "w") as file:
        file.write(_get_run_command(best_run))
    _save_model_weights(wandb_run=best_run, project=args.project, output_dirname=artifacts_dirname)


def _get_artifacts_dirname(trained_data_class: str) -> Path:
    """Return artifacts dirname."""
    for keyword in ["line", "paragraph"]:
        if keyword in trained_data_class.lower():
            artifacts_dirname = ARTIFACTS_BASE_DIRNAME / f"{keyword}_text_recognizer"
            artifacts_dirname.mkdir(parents=True, exist_ok=True)
            break
    return artifacts_dirname


def _save_model_weights(wandb_run: wandb.apis.public.Run, project: str, output_dirname: Path):
    """Save checkpointed model weights in output_dirname."""
    weights_filename = _copy_local_model_checkpoint(run_id=wandb_run.id, project=project, output_dirname=output_dirname)
    if weights_filename is None:
        weights_filename = _download_model_checkpoint(wandb_run, output_dirname)
        assert weights_filename is not None, "Model checkpoint not found"


def _copy_local_model_checkpoint(run_id: str, project: str, output_dirname: Path) -> Optional[Path]:
    """Copy model checkpoint file on system to output_dirname."""
    checkpoint_filenames = list((TRAINING_LOGS_DIRNAME / project / run_id).glob("**/*.ckpt"))
    if not checkpoint_filenames:
        return None
    shutil.copyfile(src=checkpoint_filenames[0], dst=output_dirname / "model.pt")
    print(f"Model checkpoint found on system at {checkpoint_filenames[0]}")
    return checkpoint_filenames[0]


def _download_model_checkpoint(wandb_run: wandb.apis.public.Run, output_dirname: Path) -> Optional[Path]:
    """Download model checkpoint to output_dirname."""
    checkpoint_wandb_files = [file for file in wandb_run.files() if file.name.endswith(".ckpt")]
    if not checkpoint_wandb_files:
        return None

    wandb_file = checkpoint_wandb_files[0]
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file.download(root=tmp_dirname, replace=True)
        checkpoint_filename = f"{tmp_dirname}/{wandb_file.name}"
        shutil.copyfile(src=checkpoint_filename, dst=output_dirname / "model.pt")
        print("Model checkpoint downloaded from wandb")
    return output_dirname / "model.pt"


def _get_run_command(wandb_run: wandb.apis.public.Run) -> str:
    """Return python run command for input wandb_run."""
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file = wandb_run.file("wandb-metadata.json")
        with wandb_file.download(root=tmp_dirname, replace=True) as file:
            metadata = json.load(file)

    return f"python {metadata['program']} " + " ".join(metadata["args"])


def _get_summary_value(wandb_run: wandb.apis.public.Run, key: str, default: int) -> Union[int, float]:
    """Return numeric value at summary[key] for wandb_run if it is valid, else return default."""
    value = wandb_run.summary.get(key, default)
    if not isinstance(value, (int, float)):
        value = default
    return value


def _setup_parser() -> argparse.ArgumentParser:
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", type=str, default="sergey")
    parser.add_argument("--project", type=str, default="fsdl-text-recognizer-2021-labs")
    parser.add_argument("--trained_data_class", type=str, default="IAMLines")
    parser.add_argument("--metric", type=str, default="val_loss")
    parser.add_argument("--mode", type=str, default="min")
    return parser


if __name__ == "__main__":
    save_best_model()
```
best run에 대한 정보를 정리해 `config.json`, `model.pt`, `run_command.txt`로 저장합니다. 

`_get_run_command`가 잘 동작하지 않아 그 기능을 해제한 채로 실행시켰습니다


# text_recognizer/data/iam.py
```python
"""Class for loading the IAM dataset, which encompasses both paragraphs and lines, with associated utilities."""
from pathlib import Path
from typing import Dict, List
import argparse
import os
import xml.etree.ElementTree as ElementTree
import zipfile

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.data.base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "iam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "iam"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "iamdb"

DOWNSAMPLE_FACTOR = 2  # If images were downsampled, the regions must also be.
LINE_REGION_PADDING = 16  # add this many pixels around the exact coordinates


class IAM(BaseDataModule):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.metadata = toml.load(METADATA_FILENAME)

    def prepare_data(self, *args, **kwargs) -> None:
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)  # type: ignore
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)

    @property
    def xml_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))

    @property
    def form_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))

    @property
    def form_filenames_by_id(self):
        return {filename.stem: filename for filename in self.form_filenames}

    @property
    def split_by_id(self):
        return {
            filename.stem: "test" if filename.stem in self.metadata["test_ids"] else "trainval"
            for filename in self.form_filenames
        }
```
IAM 데이터모듈을 다운로드 후 처리하면 forms 디렉토리에 이미지 파일이 저장되고, xml 디렉토리에는 xml 파일이 저장됩니다. xml은 위계적인 데이터타입이며, 트리로 나타내면 간편합니다.
glob을 이용해 해당 디렉토리에 존재하는 모든 xml파일을 `xml_filenames`라는 변수로 저장합니다. `path.stem`어트리뷰트는 파일명에서 확장자를 제외한 부분을 가져옵니다. 따라서 `a.jpg -> a`가 됩니다 


```python
    @cachedproperty
    def line_strings_by_id(self):
        """Return a dict from name of IAM form to a list of line texts in it."""
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames}

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames}

    def __repr__(self):
        """Print info about the dataset."""
        return "IAM Dataset\n" f"Num total: {len(self.xml_filenames)}\nNum test: {len(self.metadata['test_ids'])}\n"
```
`@chachedproperty` 이 데코레이터는 wrapper로 쌓인 함수가 오로지 한번 실행되고, 그 다음부터는 저장된 값을 불러오도록 합니다.

```python
def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting IAM data")
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall()
    os.chdir(curdir)


def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [el.attrib["text"].replace("&quot;", '"') for el in xml_line_elements]


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [_get_line_region_from_xml_element(el) for el in xml_line_elements]
```
 `
def _get_line_regions_from_xml_file`
xml파일을 트리 형식으로 만들어 root node를 찾고, 그 노드의 직계 자식 중 `handwritten/line`을 전부 파싱합니다.

```python
def _get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    Parameters
    ----------
    xml_line
        xml element that has x, y, width, and height attributes
    """
    word_elements = xml_line.findall("word/cmp")
    x1s = [int(el.attrib["x"]) for el in word_elements]
    y1s = [int(el.attrib["y"]) for el in word_elements]
    x2s = [int(el.attrib["x"]) + int(el.attrib["width"]) for el in word_elements]
    y2s = [int(el.attrib["y"]) + int(el.attrib["height"]) for el in word_elements]
    return {
        "x1": min(x1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "y1": min(y1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "x2": max(x2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
        "y2": max(y2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
    }


if __name__ == "__main__":
    load_and_print_info(IAM)
```
하나의 줄에 존재하는 단어들에 대한 데이터가 입력되는 상황에서 그 단어들이 차지하는 영역의 합집합, 즉 그 줄의 영역을 구하는 함수입니다. `x1s`와 `y1s1`는 단어의 시작점을 나타냅니다.(좌하단)
`x2s`는 단어의 오른쪽 끝을, `y2s`는 단어의 위쪽 끝을 나타내고, 그 값들을 downsampling(model dependent)하고 padding을 씌워 반환합니다.


# text_recognizer/data/iam_paragraphs.py
```python
"""IAM Paragraphs Dataset class."""
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.iam import IAM
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, split_dataset


PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_paragraphs"

NEW_LINE_TOKEN = "\n"
TRAIN_FRAC = 0.8

IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT = 1152 // IMAGE_SCALE_FACTOR
IMAGE_WIDTH = 1280 // IMAGE_SCALE_FACTOR
MAX_LABEL_LENGTH = 682


class IAMParagraphs(BaseDataModule):
    """
    IAM Handwriting database paragraphs.
    """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        mapping = EMNIST().mapping
        assert mapping is not None
        self.mapping = [*mapping, NEW_LINE_TOKEN]
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)  # We assert that this is correct in setup()
        self.output_dims = (MAX_LABEL_LENGTH, 1)  # We assert that this is correct in setup()

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("IAMParagraphs.prepare_data: Cropping IAM paragraph regions and saving them along with labels...")

        iam = IAM()
        iam.prepare_data()

        properties = {}
        for split in ["trainval", "test"]:
            crops, labels = get_paragraph_crops_and_labels(iam=iam, split=split)
            save_crops_and_labels(crops=crops, labels=labels, split=split)

            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[::-1],
                        "label_length": len(label),
                        "num_lines": _num_lines(label),
                    }
                    for id_, label in labels.items()
                }
            )

        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)
```
`prepare_data()`IAM datamodule의 prepare_data()를 마치고 properties라는 다중 dictionary에 crop shape의 역순, 문자열의 길이, 줄의 개수를 세서 저장합니다.
```python
    def setup(self, stage: str = None) -> None:
        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = load_processed_crops_and_labels(split)
            X = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]
            Y = convert_strings_to_labels(strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            transform = get_transform(image_shape=self.dims[1:], augment=augment)  # type: ignore
            return BaseDataset(X, Y, transform=transform)

        print(f"IAMParagraphs.setup({stage}): Loading IAM paragraph regions and lines...")
        validate_input_and_output_dimensions(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            data_trainval = _load_dataset(split="trainval", augment=self.augment)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", augment=False)
```
data와 target, transforms를 담은 `_load_dataset()`을 정의하고 self.data_train, self.data_val, self.data_test를 override합니다. transform의 경우, transform_list에 원하는 
변환을 담고, transforms.Compose로 연달아 변환이 일어나게 할 수 있습니다. 이 역할은 `_get_transforms`에서 실행되고 있습니다.
```python
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)
```
`resize_image()`이미지를 다운스케일링합니다. `resample`은 scaling의 방식을 결정합니다.
```python

def get_paragraph_crops_and_labels(iam: IAM, split: str) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Load IAM paragraph crops and labels for a given split."""
    crops = {}
    labels = {}
    for form_filename in iam.form_filenames:
        id_ = form_filename.stem
        if not iam.split_by_id[id_] == split:
            continue
        image = Image.open(form_filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)

        line_regions = iam.line_regions_by_id[id_]
        para_bbox = [
            min([_["x1"] for _ in line_regions]),
            min([_["y1"] for _ in line_regions]),
            max([_["x2"] for _ in line_regions]),
            max([_["y2"] for _ in line_regions]),
        ]
        lines = iam.line_strings_by_id[id_]

        crops[id_] = image.crop(para_bbox)
        labels[id_] = NEW_LINE_TOKEN.join(lines)
    assert len(crops) == len(labels)
    return crops, labels
```
split(fit or test)에 해당하는 이미지를 열어 grayscale로 처리하고 반전시킵니다. iam에서 얻은 line_region으로 손글씨 전체 영역(=paragraph)을 얻습니다. 잘라낸 이미지와 해당하는 손글씨를 개행 문자(\n)으로 연결한 문자열을 변수 crops와 labels로 저장하고, 반환합니다.
```python

def save_crops_and_labels(crops: Dict[str, Image.Image], labels: Dict[str, str], split: str):
    """Save crops, labels and shapes of crops of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with open(_labels_filename(split), "w") as f:
        json.dump(labels, f, indent=4)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))
```
얻은 crops는 id_.png 의 이름으로, labels는 json으로 저장합니다.  
```
def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for given split."""
    with open(_labels_filename(split), "r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels
```
sorting한 crop와 label을 불러옵니다.(잘라진 이미지입니다) convert("L")은 grayscale로 변환을 의미합니다.
```
def get_transform(image_shape: Tuple[int, int], augment: bool) -> transforms.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            transforms.RandomCrop(  # random pad image to image_shape with 0
                size=image_shape, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
            ),
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-10, 10),
                resample=Image.BILINEAR,
            ),
        ]
    else:
        transforms_list = [transforms.CenterCrop(image_shape)]  # pad image to image_shape with 0
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {"min": min(_get_property_values("num_lines")), "max": max(_get_property_values("num_lines"))},
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count("\n") + 1


if __name__ == "__main__":
    load_and_print_info(IAMParagraphs)



