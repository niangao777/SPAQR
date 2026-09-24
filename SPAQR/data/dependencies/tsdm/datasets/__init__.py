"""Dataset classes required by the four SPAQR benchmarks."""

from typing import TypeAlias

from data.dependencies.tsdm.datasets.base import (
    DATASET_OBJECT,
    BaseDataset,
    MultiFrameDataset,
    SingleFrameDataset,
)
from data.dependencies.tsdm.datasets.mimic_iii_debrouwer2019 import (
    MIMIC_III_DeBrouwer2019,
)
from data.dependencies.tsdm.datasets.physionet2012 import Physionet2012
from data.dependencies.tsdm.datasets.ushcn_debrouwer2019 import (
    USHCN_DeBrouwer2019,
)

Dataset: TypeAlias = BaseDataset

__all__ = [
    "DATASET_OBJECT",
    "BaseDataset",
    "Dataset",
    "MIMIC_III_DeBrouwer2019",
    "MultiFrameDataset",
    "Physionet2012",
    "SingleFrameDataset",
    "USHCN_DeBrouwer2019",
]
