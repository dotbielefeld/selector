"""
This module contains the default point generator.

It also contains functions to check validity of the configuration setting.
"""

from selector.pool import Configuration, ParamType


def check_conditionals(s, config_setting):
    """
    Check if conditionals are violated.

    : param s: scenario object
    : param config_setting: configuration setting to be checked
    return: List of parameter names to turn off
    """
    cond_vio = []
    for condition in s.conditionals:
        for cond in s.conditionals[condition]:
            for param in s.parameter:
                if param.name == cond:
                    if param.type == ParamType.categorical and \
                            (config_setting[cond] not in
                             s.conditionals[condition][cond] and
                             condition not in cond_vio and
                             condition in config_setting):
                        cond_vio.append(condition)
                    elif (param.type == ParamType.continuous or
                            param.type == ParamType.integer) and \
                            (config_setting[cond] >
                             s.conditionals[condition][cond][1] or
                             config_setting[cond] <
                             s.conditionals[condition][cond][0]) and \
                            condition not in cond_vio and \
                            condition in config_setting:
                        cond_vio.append(condition)

    return cond_vio


def check_no_goods(s, config_setting):
    """
    Check if conditionals are violated.

    : param s: scenario object
    : param config_setting: configuration setting to be checked
    return: True/False, if no goods are violated
    """
    check = False
    for ng in s.no_goods:
        param_1, param_2 = ng.keys()
        check = (not check and ng[param_1] == config_setting[param_1]) \
            and (ng[param_2] == config_setting[param_2])

    return check


def default_point(s, identity):
    """
    Default parameter setting is generated in Configuration format.

    : param s: scenario object
    : param identity: uuid to identify configuration
    return: default configuration
    """
    config_setting = {}

    # Generate configuration with default values
    for param in s.parameter:
        config_setting[param.name] = param.default

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity, config_setting)

    return configuration
