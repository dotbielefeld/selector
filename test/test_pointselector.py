"""This module contains simple tests for the point selection functions."""
import unittest
import numpy as np
import ray
import pickle
from selector.ta_result_store import TargetAlgorithmObserver
from selector.pool import Configuration, Generator
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.scenario import Scenario, parse_args


class RandomPointselectorTest(unittest.TestCase):
    """Testing Random_Pointselector_Test."""

    def setUp(self):
        """Set up unittest."""
        self.pool = {}
        test_confs = [Configuration(1, {"param_1": 10, "param_2": 20},
                                    Generator.random),
                      Configuration(2, {"param_1": 50, "param_2": 30},
                                    Generator.random),
                      Configuration(3, {"param_1": 20, "param_2": 40},
                                    Generator.random)]
        for conf in test_confs:
            self.pool[conf.id] = conf

    def test_random_pointselector(self):
        """Testing random point selector."""
        iteration = 1
        np.random.seed(42)
        selector = RandomSelector()
        selected_ids = selector.select_points(self.pool, 2, iteration,
                                              seed=42)
        # selected are [1, 2]
        self.assertEqual(selected_ids[0], 1)
        self.assertEqual(selected_ids[1], 2)


class HyperparameterizedSelectorTest(unittest.TestCase):
    """Testing HyperparameterizedSelector."""

    def setUp(self):
        """Set up unittest."""
        file = open('./test/s', 'rb')
        self.s = pickle.load(file)
        file.close()
        self.random_generator = PointGen(self.s, random_point, seed=42)
        # Set up a tournament with mock data
        global_cache = TargetAlgorithmObserver.remote()
        point_selector = RandomSelector()
        for i in range(5):
            instance_selector = InstanceSet(self.s.instance_set, 1, 1)
            generated_points = [self.random_generator.point_generator(
                                seed=42 + i)
                                for _ in range(self.s.tournament_size *
                                               self.s.generator_multiple)]
            points_to_run = point_selector.select_points(
                generated_points,
                self.s.tournament_size, 0)
            instance_id, instances = instance_selector.get_subset(0)

            tourn, _ = MiniTournamentDispatcher().init_tournament(
                global_cache,
                points_to_run,
                instances,
                instance_id)

            tourn.best_finisher = [tourn.configurations[i]]

            tourn.configurations.pop(i)
            tourn.worst_finisher = tourn.configurations
            tourn.configurations = []
            global_cache.put_tournament_history.remote(tourn)

        self.hist = ray.get(global_cache.get_tournament_history.remote())
        self.default_generator = PointGen(self.s, default_point, seed=42)
        self.variable_graph_generator = PointGen(self.s, variable_graph_point,
                                                 seed=42)
        self.lhc_generator = PointGen(self.s, lhc_points, seed=42)

    def test_hyperparameterized_pointselector(self):
        """Testing hyperparameterized point selector."""
        def_conf = [self.default_generator.point_generator()]

        print(self.s)

        ran_conf = []
        for i in range(5):
            ran_conf.append(self.random_generator.point_generator(seed=42 + i))

        var_conf = []

        for i in range(4):
            var_conf.append(self.variable_graph_generator.point_generator(
                mode=Mode.random,
                data=self.hist, lookback=i + 1, seed=(42 + i)))

        lhc_conf = \
            self.lhc_generator.point_generator(n_samples=5, seed=42,
                                               lhs_type=LHSType.centered,
                                               criterion=Criterion.maximin)

        confs = def_conf + ran_conf + var_conf + lhc_conf

        hps = HyperparameterizedSelector()

        configs_requested = 8
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        weights = [weights for x in range(len(confs))]
        weights = np.array(weights)
        epoch = 4
        max_epochs = 256

        selected_ids = []

        for epoch in range(2):
            selected_ids.append(hps.select_points(self.s, confs,
                                                  configs_requested, epoch,
                                                  max_epochs, weights,
                                                  max_evals=100, seed=42))

        self.assertEqual(selected_ids[0], selected_ids[1])

if __name__ == '__main__':
    unittest.main()
