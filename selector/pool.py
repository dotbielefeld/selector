from dataclasses import dataclass, field

from typing import Dict


@dataclass
class Configuration:
    id: int
    conf: dict
    gender: str

@dataclass
class Parameter:
    name: str
    type: str
    bound: list
    default: int
    condition: list
    scale: str


def init_pool():
    pass
