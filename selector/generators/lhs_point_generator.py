"""This module contains the latin hyper cube graph point generator."""

from skopt.space import Space
from skopt.sampler import Lhs
from enum import Enum
import uuid
from selector.pool import Configuration, ParamType, Generator
from selector.generators.default_point_generator import (
    check_no_goods
)
from selector.generators.random_point_generator import reset_no_goods


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
    Generate the sampling space for the Latin Hypercube (LHC) according to scenario parameters.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.

    Returns
    -------
    skopt.Space object
        Latin Hypercube space.
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

    Parameters
    ----------
    space : skopt.Space object
        Sampling space for the Latin Hypercube (LHC).
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed; will be set if not False.
    lhs_type : 
        Sampling type parameter for `skopt.sampler.Lhs`.
    criterion : 
        Optimization criterion for `skopt.sampler.Lhs`.

    Returns
    -------
    list of dict
        Generated n samples.
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
    Generate configuration using the variable graph method.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    identity : uuid.UUID
        This UUID is just a placeholder.
    n_samples : int
        Number of picks from the parameter space.
    seed: int
        Random seed.
    lhs_type: selector.generators.lhs_point_generator.LHSType
        Type of LHC sampling.
    criterion: selector.generators.lhs_point_generator.Criterion
        Criterions of LHC optimizatio.
    Returns
    -------
    list of selector.pool.Configuration
        List of generated configurations.
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
