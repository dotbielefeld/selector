import numpy as np


class PointSelector:

    def __init__(self):
        self.selection_history = {}

    def select_points(self, pool, number_of_points, iteration):
        pass


class RandomSelector(PointSelector):

    def __init__(self):
        super().__init__()

    def select_points(self, pool, number_of_points, iteration):
        """
        Randomly select a subset of configurations from the pool to run
        :param pool: dic. Pool of configurations to select from
        :param number_of_points: int. Number of points to select from the pool.
        :param iteration: int. Iteration identifier which stores the selection for later reference
        :return: list. Ids of configurations from pool that are selected
        """

        selected_ids = np.random.choice(list(pool), number_of_points, replace=False)
        self.selection_history[iteration] = selected_ids

        return selected_ids

