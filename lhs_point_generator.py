"""This module contains the latin hyper cube graph point generator."""

from skopt.space import Space
from skopt.sampler import Lhs
from selector.pool import Configuration, ParamType
from selector.default_point_generator import check_conditionals, check_no_goods


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


def lhc_points(s, identity, n_samples=1):
    """
    Configuration is generated via variable graph method.

    : param s: scenario object
    : param identity: uuid to identify configuration
    : param n_samples: int, number of picks from parameter space
    return: n configurations
    """
    space = generate_space(s)

    lhs = Lhs(lhs_type="classic", criterion=None)

    n_points = lhs.generate(space.dimensions, n_samples)

    names = []

    for param in s.parameter:
        names.append(param.name)

    n_configurations = []

    for conf in n_points:
        conf_values = []
        for value in conf:
            conf_values.append(value)
        zip_it = zip(names, conf_values)
        config = dict(zip_it)
        n_configurations.append(Configuration(identity, config))

    return n_configurations
