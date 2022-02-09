from selector.pointselector import RandomSelector
import unittest
from selector.pool import Configuration
import numpy as np

np.random.seed(42)

class Random_Pointselector_Test(unittest.TestCase):

    def setUp(self):

        self.pool = {}
        test_confs = [Configuration(1, {"param_1": 10, "param_2": 20}), Configuration(2, {"param_1": 50, "param_2": 30}),
                      Configuration(3, {"param_1": 20, "param_2": 40})]
        for conf in test_confs:
            self.pool[conf.id] = conf


    def test_random_pointselector(self):
        iteration = 1
        selector = RandomSelector()
        selected_ids = selector.select_points(self.pool, 2, iteration)
        # selected are [3, 1]
        self.assertEqual(selected_ids[0], 3)
        self.assertEqual(selected_ids[1], 1)
