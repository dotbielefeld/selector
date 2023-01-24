"""This module contains simple tests for the point selection functions."""
import unittest
import numpy as np
import pickle
from selector.pool import (
    Configuration,
    Generator,
    Tournament,
    Surrogates
)
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import default_point
from selector.generators.variable_graph_point_generator import (
    variable_graph_point,
    Mode
)
from selector.generators.lhs_point_generator import (
    lhc_points,
    LHSType,
    Criterion
)
from selector.selection_features import FeatureGenerator
from selector.generators.surrogates.surrogates import SurrogateManager
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
        file = open('./test/scenario', 'rb')
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
                self.results[conf.id] = \
                    {'./test/test_data/instances/' +
                     'test_instance_1.cnf': np.random.randint(2, 15),
                     './test/test_data/instances/' +
                     'test_instance_2.cnf': np.random.randint(2, 15)}
            else:
                self.results[conf.id] = \
                    {'./test/test_data/instances/' +
                     'test_instance_1.cnf': np.random.randint(2, 15)}

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
                           ['./test/test_data/instances/test_instance_1.cnf'],
                           0)

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
                results=self.results, mode=Mode.random,
                alldata=self.hist, lookback=i + 1, seed=(42 + i)))

        lhc_conf = \
            self.lhc_generator.point_generator(n_samples=5, seed=42,
                                               lhs_type=LHSType.centered,
                                               criterion=Criterion.maximin)

        sm = SurrogateManager(self.s, seed=42)
        smac_conf = sm.suggest(Surrogates.SMAC, self.s,
                               5, None, None, None)

        ggapp_conf = sm.suggest(Surrogates.GGApp, self.s, 5, self.hist,
                                self.results, None)
        cppl_conf = \
            sm.suggest(Surrogates.CPPL, self.s, 5, None, None,
                       ['./test/test_data/instances/test_instance_1.cnf'])[0]

        confs = def_conf + ran_conf + var_conf + lhc_conf + smac_conf + \
            ggapp_conf + cppl_conf

        hps = HyperparameterizedSelector()

        configs_requested = 8
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        weights = [weights for x in range(len(confs))]
        weights = np.array(weights)
        epoch = 4
        max_epochs = 256

        selected_ids = []
        predicted_quals = []
        evaluated = []

        fg = FeatureGenerator()

        cutoff_time = float(self.s.cutoff_time)
        predicted_perf = []
        predicted_quals = []
        qap = False
        evaluated = []

        for epoch in range(2):

            result_tournament = self.hist[list(self.hist.keys())[4]]

            result_tournament.configuration_ids = \
                [result_tournament.configuration_ids[0]]

            all_configs \
                = result_tournament.best_finisher \
                + result_tournament.worst_finisher

            terminations = []

            for surrogate in sm.surrogates.keys():
                sm.update_surr(surrogate, result_tournament, all_configs,
                               self.results, terminations)

            features = fg.static_feature_gen(confs, epoch, max_epochs)
            features \
                = np.concatenate((features,
                                  fg.diversity_feature_gen(confs, self.hist,
                                                           self.results,
                                                           cutoff_time,
                                                           self.s.parameter,
                                                           predicted_quals,
                                                           evaluated)),
                                 axis=1)

            instances = ['./test/test_data/instances/test_instance_1.cnf']

            for surrogate in sm.surrogates.keys():
                if surrogate is Surrogates.SMAC:
                    if sm.surrogates[surrogate].surr.model.rf is not None:
                        predicted_perf = sm.predict(surrogate,
                                                    confs,
                                                    cutoff_time, None)
                else:
                    predicted_perf = sm.predict(surrogate,
                                                confs,
                                                cutoff_time,
                                                instances)

            features = np.concatenate((features,
                                      fg.dynamic_feature_gen(confs, self.hist,
                                                             predicted_perf,
                                                             sm, cutoff_time,
                                                             self.results,
                                                             instances)),
                                      axis=1)

            selected_ids = hps.select_points(self.s, confs, configs_requested,
                                             epoch, max_epochs, features,
                                             weights, self.results,
                                             max_evals=100, seed=42)

            for surrogate in sm.surrogates.keys():
                if surrogate is Surrogates.SMAC:
                    if sm.surrogates[surrogate].surr.model.rf is not None:
                        if qap:
                            predicted_quals.extend(sm.predict(surrogate,
                                                              selected_ids,
                                                              cutoff_time,
                                                              None))
                        else:
                            predicted_quals.extend(sm.predict(surrogate,
                                                              evaluated,
                                                              cutoff_time,
                                                              None))
                            qap = True

                else:
                    if qap:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          selected_ids,
                                                          cutoff_time,
                                                          instances))
                    else:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          evaluated,
                                                          cutoff_time,
                                                          instances))
                        qap = True

            evaluated.extend(selected_ids)

            for conf in selected_ids:
                if selected_ids[-1] == conf:
                    self.results[conf.id] = {1: np.random.randint(2, 15),
                                             0: np.random.randint(2, 15)}
                else:
                    self.results[conf.id] = {1: np.random.randint(2, 15)}

            ran_conf = []
            for i in range(5):
                ran_conf.append(
                    self.random_generator.point_generator(seed=42 + i))

            var_conf = []

            for i in range(5):
                var_conf.append(self.variable_graph_generator.point_generator(
                    results=self.results, mode=Mode.random,
                    alldata=self.hist, lookback=i + 1, seed=(42 + i)))

            lhc_conf \
                = self.lhc_generator.point_generator(n_samples=5, seed=42,
                                                     lhs_type=LHSType.centered,
                                                     criterion=Criterion.
                                                     maximin)

            smac_conf = sm.suggest(Surrogates.SMAC, self.s, 5,
                                   None, None, None)

            ggapp_conf = sm.suggest(Surrogates.GGApp, self.s, 5, self.hist,
                                    self.results, None)
            cppl_conf = \
                sm.suggest(Surrogates.CPPL, self.s, 5, None, None,
                           instances)[0]

            confs = ran_conf + var_conf + lhc_conf + smac_conf + \
                ggapp_conf + cppl_conf

        test_1 = Configuration(1,
                               {'luby': True, 'rinc': 3.1300000000000003,
                                'cla-decay': 0.909999,
                                'phase-saving': 1, 'bce-limit': 60070000,
                                'param_1': -2, 'strSseconds': '100'},
                               Generator.var_graph)

        self.assertEqual(selected_ids[1].conf, test_1.conf)

if __name__ == '__main__':
    unittest.main()
