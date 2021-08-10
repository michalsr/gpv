from typing import List

from allennlp.common import Registrable
from torch.utils.data import Dataset as TorcDataset

from exp.ours.data.gpv_data import GPVExample
import numpy as np


class DatasetBuilder(Registrable):

  def build(self, examples: List[GPVExample]) -> TorcDataset:
    # Note this might be run on multiple processes, so it should be deterministic,
    raise NotImplementedError()
