"""This module contains simple tests for the point selection functions."""
import unittest
import numpy as np
import pickle
from selector.pool import Configuration, Generator, Tournament
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator
# from selector.surrogates.surrogates import SurrogateManager
import uuid
import copy


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
        point_selector = RandomSelector()

        generated_points = [self.random_generator.point_generator(
                            seed=42)
                            for _ in range(self.s.tournament_size)]
        point_selector.select_points(
            generated_points,
            self.s.tournament_size, 0)

        self.results = {}

        for conf in generated_points:
            if generated_points[-1] == conf:
                self.results[conf.id] = {0: np.random.randint(2, 15),
                                         1: np.random.randint(2, 15)}
            else:
                self.results[conf.id] = {0: np.random.randint(2, 15)}

        self.hist = {}

        for i in range(5):
            gp = copy.deepcopy(generated_points)
            tourn_id = uuid.uuid4()
            best_finisher = np.random.choice(gp)
            index = gp.index(best_finisher)
            del gp[index]
            worst_finisher = np.random.choice(gp, size=5).tolist()
            configuration_ids = []
            for conf in generated_points:
                configuration_ids.append(conf.id)
            self.hist[tourn_id] = \
                Tournament(tourn_id, [best_finisher], worst_finisher,
                           [], configuration_ids, {},
                           ['instance_1.cnf'], 0)

        self.default_generator = PointGen(self.s, default_point, seed=42)
        self.variable_graph_generator = PointGen(self.s, variable_graph_point,
                                                 seed=42)
        self.lhc_generator = PointGen(self.s, lhc_points, seed=42)

    def test_hyperparameterized_pointselector(self):
        """Testing hyperparameterized point selector."""
        def_conf = [self.default_generator.point_generator()]

        cutoff_time = float(self.s.cutoff_time)

        ran_conf = []
        for i in range(5):
            ran_conf.append(self.random_generator.point_generator(seed=42 + i))

        var_conf = []

        for i in range(4):
            var_conf.append(self.variable_graph_generator.point_generator(
                mode=Mode.random,
                alldata=self.hist, lookback=i + 1, seed=(42 + i)))

        lhc_conf = \
            self.lhc_generator.point_generator(n_samples=5, seed=42,
                                               lhs_type=LHSType.centered,
                                               criterion=Criterion.maximin)

        confs = def_conf + ran_conf + var_conf + lhc_conf

        hps = HyperparameterizedSelector()

        configs_requested = 8
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        weights = [weights for x in range(len(confs))]
        weights = np.array(weights)
        epoch = 4
        max_epochs = 256

        selected_ids = []
        predicted_quals = []
        evaluated = []

        fg = FeatureGenerator()
        # sm = SurrogateManager(self.s.parameter)

        for epoch in range(2):

            features = fg.static_feature_gen(confs, epoch, max_epochs)
            features = np.concatenate(
                (features, fg.diversity_feature_gen(confs, self.hist,
                                                    self.results, cutoff_time,
                                                    self.s.parameter,
                                                    predicted_quals,
                                                    evaluated)),
                axis=1)
            '''
            features = np.concatenate((features,
                                      fg.dynamic_feature_gen(confs, self.hist,
                                                             predicted_quals,
                                                             sm, cutoff_time,
                                                             self.s.parameter,
                                                             self.results)),
                                      axis=1)
            '''

            selected_ids.append(hps.select_points(self.s, confs,
                                                  configs_requested, epoch,
                                                  max_epochs, features,
                                                  weights, self.results,
                                                  max_evals=100, seed=42)[0])

            evaluated.extend(selected_ids)

            '''
            predicted_quals.extend(sm.expected_value(selected_ids,
                                                     self.s.parameter,
                                                     cutoff_time,
                                                     surrogate='SMAC'))
            '''

            for conf in selected_ids:
                if selected_ids[-1] == conf:
                    self.results[conf.id] = {1: np.random.randint(2, 15),
                                             0: np.random.randint(2, 15)}
                else:
                    self.results[conf.id] = {1: np.random.randint(2, 15)}

            '''
            for conf in selected_ids:
                for surrogate in sm.surrogates.keys():
                    sm.observe(conf.conf, self.results[conf.id][epoch],
                               self.s.parameter, cutoff_time, surrogate)
            '''

        test_1 = Configuration(1,
                               {'luby': False, 'rinc': 1.3900000000000001,
                                'cla-decay': 0.9299970000000001,
                                'phase-saving': 2, 'bce-limit': 20090000,
                                'param_1': -1},
                               Generator.var_graph)

        self.assertEqual(selected_ids[0].conf, test_1.conf)

if __name__ == '__main__':
    unittest.main()
