"""This module contains the point generation class."""

import uuid
from selector.random_point_generator import random_point


class PointGen:
    """Interface for point generation."""

    def __init__(self, scenario, gm=random_point):
        """
        Initialize PointGen.

        : param scenario: scenario object
        : param gm: point generating method to use
        """
        self.s = scenario
        self.id = uuid.uuid4()
        self.gen_method = gm

    def point_generator(self):
        """
        Running point generation according to object setting.

        return: configuration/point generated
        """
        configuration = self.gen_method(self.s, self.id)

        return configuration
