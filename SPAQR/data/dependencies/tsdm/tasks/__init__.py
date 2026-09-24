"""Forecasting tasks required by the four SPAQR benchmarks."""

from typing import TypeAlias

from data.dependencies.tsdm.tasks.base import BaseTask
from data.dependencies.tsdm.tasks.mimic_iii_debrouwer2019 import (
    MIMIC_III_DeBrouwer2019,
)
from data.dependencies.tsdm.tasks.P12 import Physionet2012
from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import (
    USHCN_DeBrouwer2019,
)

Task: TypeAlias = BaseTask

__all__ = [
    "BaseTask",
    "MIMIC_III_DeBrouwer2019",
    "Physionet2012",
    "Task",
    "USHCN_DeBrouwer2019",
]
