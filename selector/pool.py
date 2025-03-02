"""In this module the most important data structures are defined."""
from dataclasses import dataclass
from enum import Enum


@dataclass
class Configuration:
    """
    Configuration data object.
    
    Parameters
    ----------
    id : int
        uuid.UUID of the Configuration
    conf : dict
        {Parameter name : parameter value}
    generator : selector.pool.Generator
        Identifies how the configuration was generated.
    """
    id: int
    conf: dict
    generator: Enum


@dataclass
class Parameter:
    """
    Parameter data object.

    Parameters
    ----------
    name : str
        Parameter name.
    type : str
        Parameter type
    bound : list
        Upper and lower bound, or list of possible values.
    default : int
        Index of the default value in the bound, or float.
    condition : list
        List of conditional values for this parameter.
    scale : str
        Linear or log.
    original_bound : list
        Unprocessed bound (as read in from .pcs).
    """
    name: str
    type: str
    bound: list
    default: int
    condition: list
    scale: str
    original_bound: list


@dataclass
class Tournament:
    """
    Tournament data object.

    Parameters
    ----------
    name : str
        Parameter name.
    best_finisher : list
        Winner(s) of the tournament.
    worst_finisher : list
        The rest of the participants.
    configurations : list
        List of configuration IDs.
    ray_object_store : dict
        Ray object adresses.
    instance_set : list
        Tournament instance names.
    instance_set_id : int
        ID of the instance set.
    """
    id: int
    best_finisher: list
    worst_finisher: list
    configurations: list
    configuration_ids: list
    ray_object_store: dict
    instance_set: list
    instance_set_id: int


class ParamType(Enum):
    """Parameter type enumerator."""
    categorical = 1
    continuous = 2
    integer = 3


class TaskType(Enum):
    """Task type enumerator."""
    target_algorithm = 1
    monitor = 2


class Generator(Enum):
    """Generator enumerator."""
    default = 1
    random = 2
    var_graph = 3
    lhc = 4
    smac = 5
    ggapp = 6
    cppl = 7
    base = 8


class Status(Enum):
    """Status enumerator."""
    win = 1
    cap = 2
    timeout = 3
    stop = 4
    running = 5


class Surrogates(Enum):
    """Surrogate type enumerator."""
    SMAC = 1
    GGApp = 2
    CPPL = 3
