"""This module contains functions for selection of points."""
import numpy as np
from selector.hp_point_selection import select_point


class PointSelector:
    """Generic point selector class."""

    def __init__(self, features=None):
        """Initialize class."""
        self.selection_history = {}
        self.features = features

    def select_points(self, pool, number_of_points, iteration):
        """Generic point selector method."""
        pass


class RandomSelector(PointSelector):
    """Random point selector class."""

    def __init__(self):
        """Initialize class."""
        super().__init__()

    def select_points(self, pool, number_of_points, iteration, seed=False):
        """
        Randomly select a subset of configurations from the pool to run.

        :param pool: dic. Pool of configurations to select from
        :param number_of_points: int. Number of points to select from the pool.
        :param iteration: int. Iteration identifier which stores the selection
                          for later reference
        :return: list. Ids of configurations from pool that are selected
        """
        if seed:
            np.random.seed(seed)
        selected_points = np.random.choice(list(pool), number_of_points,
                                           replace=False)
        self.selection_history[iteration] = selected_points

        return selected_points.tolist()


class HyperparameterizedSelector(PointSelector):
    u"""
    Hyperparameterized selection of generated points.

    Based on:
    Carlos Ansótegui, Meinolf Sellmann, Tapan Shah,
    Kevin Tierney,
    Learning to Optimize Black-Box Functions With
    Extreme Limits on the Number of Function Evaluations,
    2021, International Conference on Learning and Intelligent
    Optimization, 7-24
    """

    def __init__(self):
        """Initialize class."""
        super().__init__()

    def select_points(self, scenario, pool, number_of_points, epoch,
                      max_epoch, features, weights, results, max_evals=100,
                      seed=False):
        """
        Select configurations subset from pool based on scoring function.

        :param pool: dic. Pool of configurations to select from
        :param number_of_points: int. Number of points to select from the pool.
        :param iteration: int. Iteration identifier which stores the selection
                          for later reference
        :param MAX_EPOCH: int. How many simulations per selecte point
        :return: list. Ids of configurations from pool that are selected
        """
        selected_points = select_point(scenario, list(pool), max_evals,
                                       number_of_points, pool, epoch,
                                       max_epoch, features, weights, seed)

        self.selection_history[epoch] = selected_points

        return selected_points
