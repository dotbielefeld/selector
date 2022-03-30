from dataclasses import dataclass, field
import random
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

@dataclass
class Tournament:
    id : int
    best_finisher: list
    worst_finisher: list
    configurations: list
    configuration_ids: list
    ray_object_store: dict
    instance_set: list
    instance_set_id : int

class ParamType(Enum):
    categorical = 1
    continuous = 2
    integer = 3


class TaskType(Enum):
    target_algorithm = 1
    monitor = 2




