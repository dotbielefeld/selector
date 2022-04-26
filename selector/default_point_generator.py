"""This module contains the default point generator."""

from selector.pool import Configuration


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
