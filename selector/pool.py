from dataclasses import dataclass, field

from typing import Dict
from enum import Enum


@dataclass
class Configuration:
    id: int
    conf: dict

@dataclass
class Parameter:
    name: str
    type: str
    bound: list
    default: int
    condition: list
    scale: str

class ParamType(Enum):
    categorical = 1
    continuous = 2
    integer = 3


def init_pool():
    pass
