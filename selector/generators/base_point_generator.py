"""This module contains the base point generation."""

from selector.pool import Configuration, Generator
import uuid


def base_point(s, identity, seed=False):
    """
    Random parameter setting is generated in Configuration format.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    identity : uuid.UUID
        UUID to identify configuration.

    Returns
    -------
    dict
        Configuration.
    """
    configuration = \
        Configuration(uuid.uuid4(), {'meaning': 42}, Generator.base)

    return configuration
