"""This module contains the latin hyper cube graph point generator."""

from skopt.space import Space
from skopt.sampler import Lhs
from enum import Enum
import uuid
from selector.pool import Configuration, ParamType, Generator
from selector.generators.default_point_generator import (
    check_conditionals,
    check_no_goods
)
from selector.generators.random_point_generator import reset_no_goods,\
    reset_conditionals


class LHSType(Enum):
    """Contains the types of LHS."""

    classic = 1
    centered = 2


class Criterion(Enum):
    """Contains the criterions of optimization of LHS."""

    correlation = 1
    maximin = 2
    ratio = 3


def generate_space(s):
    """
    Generating the sampling space for the lhc according to scenario parameters.

    : param scenario: scenario object
    """
    space_list = []
    for ps in s.parameter:
        if ps.type == ParamType.categorical:
            # categorical space defined by list
            space_list.append(ps.bound)
        else:
            # int/real space defined by tuple
            space_list.append(tuple(ps.bound))

    space = Space(space_list)

    return space


def get_n_points(space, n_samples, seed, lhs_type, criterion):
    """
    Generate n samples.

    : param space: sampling space for the lhc
    : param n_samples: number of samples to generate
    : param seed: will set random seed, if not False
    : param lhs_type: sampling type parameter for skopt.sampler.Lhs
    : param criterion: optimization criterion for skopt.sampler.Lhs
    return: n samples
    """
    if lhs_type == LHSType.centered:
        lt = 'centered'
    else:
        lt = 'classic'

    if criterion == Criterion.correlation:
        cr = 'correlation'
    elif criterion == Criterion.maximin:
        cr = 'maximin'
    elif criterion == Criterion.ratio:
        cr = 'ratio'
    else:
        cr = None

    lhs = Lhs(lhs_type=lt, criterion=cr)

    if seed:
        n_samples = lhs.generate(space.dimensions, n_samples,
                                 random_state=seed)
    else:
        n_samples = lhs.generate(space.dimensions, n_samples)

    return n_samples


def lhc_points(s, identity, n_samples=1, seed=False, lhs_type=LHSType.classic,
               criterion=None):
    """
    Configuration is generated via variable graph method.

    : param s: scenario object
    : param identity: uuid to identify configuration
    : param n_samples: int, number of picks from parameter space
    return: n configurations
    """
    space = generate_space(s)

    n_samples = get_n_points(space, n_samples, seed, lhs_type, criterion)

    param_names = []
    for param in s.parameter:
        param_names.append(param.name)

    n_points = []
    for sample in n_samples:
        point = {}
        for i in range(len(sample)):
            point[param_names[i]] = sample[i]
        n_points.append(point)

    # Check no goods and reset values if violated
    for point in n_points:
        ng_vio = check_no_goods(s, point)
        while ng_vio:
            point = reset_no_goods(s, point)
            ng_vio = check_no_goods(s, point)

    n_configurations = []

    if len(n_points) > 1:
        mult = True

    for conf in n_points:
        if mult:
            identity = uuid.uuid4()
        n_configurations.append(Configuration(identity, conf, Generator.lhc))

    return n_configurations
