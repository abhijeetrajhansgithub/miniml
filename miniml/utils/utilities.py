from charset_normalizer import from_path                              # type: ignore
import csv
import ast
import json
import re
import math
from typing import Any, Dict, List, Tuple, cast, Optional             # type: ignore

from miniml._memory._memory import MemoryProfile
from miniml.errors.errors import DatasetInferencingError

def get_encoding(file_path: str) -> str:
    return from_path(file_path).best().encoding                       # type: ignore


def get_delimiter(file_path: str, sample_size: int = 5000) -> str:
    with open(file_path, "r") as f:
        dialect = csv.Sniffer().sniff(f.read(sample_size))
        return dialect.delimiter
