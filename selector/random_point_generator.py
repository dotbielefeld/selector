"""This module contains functions for random point generation."""

import numpy as np
import math
from selector.pool import Configuration, ParamType


def remove_infeasible(s, config_setting):
    """
    Checking Conditionals and turning parameters off if violated.

    : param s: scenario object
    : param config_setting: parameter configuration
    return: config_estting adjusted to conditionals
    """
    conf_to_del = []

    # Child node is turned off if parent node does
    # not take value in specified range
    for child_node in s.conditionals:
        for parent_node in s.conditionals[child_node]:
            for params in s.parameter:
                if params.name \
                        == parent_node:

                    parent_info = s.conditionals[child_node]
                    param_space = params

                    continue

            for parent in parent_info:
                if param_space.type == ParamType.categorical:
                    if config_setting[parent_node] \
                            not in parent_info[parent_node]:
                        if child_node not in conf_to_del:
                            conf_to_del.append(child_node)

                elif param_space.type == ParamType.continuous or \
                        param_space.type == ParamType.integer:
                    # Is the parameter value in the range?
                    if parent_info[parent_node][0] \
                        > config_setting[parent_node] or \
                            config_setting[parent_node] \
                            > parent_info[parent_node][1]:

                        if child_node not in conf_to_del:
                            conf_to_del.append(child_node)

        for ctd in conf_to_del:
            config_setting.pop(ctd, None)

    return config_setting


def satisfy_no_goods(s, config_setting):
    """
    Checking for no goods and resetting parameter values if violated.

    : param s: scenario object
    : param config_setting: parameter configuration
    return: config_estting adjusted to no goods
    """
    for ng in s.no_goods:

        violation = True
        for i in range(10):
            if violation:
                ng_values = list(ng.values())
                config_set_values = []

                for ng_element in ng:
                    config_set_values.append(config_setting[ng_element])

                if config_set_values == ng_values:
                    configs_to_reset = []
                    violation = True

                    for params in s.parameter:
                        if params.name in ng:
                            configs_to_reset.append(params)

                    new_setting = random_set_conf(configs_to_reset)

                    for ns in new_setting:
                        config_setting[ns] = new_setting[ns]

                else:
                    violation = False
            else:
                continue

    return config_setting


def random_set_conf(parameter):
    """
    Generating random configuration values for given param space.

    : param parameter: dataclass Parameter, filled out with scenario data
    return: randomly set parameters
    """
    config_setting = {}

    for param in parameter:

        if param.type == ParamType.categorical:
            config_setting[param.name] = np.random.choice(param.bound)

        elif param.type == ParamType.continuous:
            if param.scale:
                # Generate in logarithmic space
                config_setting[param.name] \
                    = math.exp(np.random.uniform(low=math.log(param.bound[0]),
                                                 high=(param.bound[1])))
            else:
                config_setting[param.name] \
                    = np.random.uniform(low=param.bound[0],
                                        high=param.bound[1])

        elif param.type == ParamType.integer:
            if param.scale:

                # Generate in logarithmic space
                config_setting[param.name] \
                    = int(math.exp(np.random.randint(
                          low=math.log(param.bound[0]),
                          high=math.log(param.bound[1]))))
            else:
                config_setting[param.name] \
                    = np.random.randint(low=param.bound[0],
                                        high=param.bound[1])

    return config_setting


def random_point(s, identity):
    """
    Random parameter setting is generated in Configuration format.

    : param s: scenario object
    : param identity: uuid to identify configuration
    return: randomly set configuration, which accounts for no goods
    and conditionals
    """
    # Generate configuration randomly based on given parameter space
    config_setting = random_set_conf(s.parameter)

    # Check conditionals and turn off parameters if violated
    config_setting = remove_infeasible(s, config_setting)

    # Check no goods and reset values if violated
    config_setting = satisfy_no_goods(s, config_setting)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity, config_setting)

    return configuration
