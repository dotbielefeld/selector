"""This module contains functions for selection of points."""
import numpy as np
from selector.hp_point_selection import select_point


class PointSelector:
    """
    Generic point selector class.

    Parameters
    ----------
    features : list of dict
        Problem instance features.
    """

    def __init__(self, features=None):
        self.selection_history = {}
        self.features = features

    def select_points(self):
        """Generic point selector method."""
        pass


class RandomSelector(PointSelector):
    """Random point selector class."""

    def __init__(self):
        super().__init__()

    def select_points(self, pool, number_of_points, iteration, seed=False):
        """
        Randomly select a subset of configurations from the pool to run.

        Parameters
        ----------
        pool : dict
            Pool of configurations to select from.
        number_of_points : int
            Number of points to select from the pool.
        iteration : int
            Iteration identifier which stores the selection for later reference.
        seed : int
            Random seed.

        Returns
        -------
        list
            IDs of configurations from the pool that are selected.
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

    Note
    ----
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
        Select a subset of configurations from the pool based on a scoring function.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        pool : dict
            Pool of configurations to select from.
        number_of_points : int
            Number of points to select from the pool.
        epcoch : int
            Iteration identifier which stores the selection for later reference.
        max_epoch : int
            Maximum number of iterations for the AC process (meaningless if termination criterion is total_runtime).
        features : ndayrray
            Configuration features computed for each configuration in the pool.
        weights : ndarray
            Pre-computed/ set weights for the scoring fuction.
        results : dict
            Results for configuration /instance pairs.
        max_evals : int
            Number of simulations per selected point.
        seed : int
            Random seed.

        Returns
        -------
        list
            IDs of configurations from the pool that are selected.
        """
        selected_points = select_point(scenario, list(pool), max_evals,
                                       number_of_points, pool, epoch,
                                       max_epoch, features, weights, seed)

        self.selection_history[epoch] = selected_points

        return selected_points
