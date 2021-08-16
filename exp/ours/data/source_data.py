"""Functional interface to our "raw" data"""

import logging
import sys
from collections import defaultdict, Counter
from os.path import join, exists, dirname
from typing import List, Optional, Any, Dict, Tuple

from dataclasses import dataclass

from exp.ours import file_paths
from exp.ours.data.dataset import LocalizationExample, VqaExample, ClsExample, CaptioningExample, \
  Caption, Task
from exp.ours.util import image_utils
from utils.io import load_json_object, dump_json_object
import numpy as np

