"""This module contains simple tests for the point generation functions."""
import ray
import unittest
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import default_point
from selector.generators.variable_graph_point_generator import variable_graph_point, Mode
from selector.generators.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.scenario import Scenario, parse_args
from selector.generators.surrogates.surrogates import SurrogateManager
from selector.pool import Surrogates, Tournament
import pickle
import uuid
import copy
import numpy as np


class PointGenTest(unittest.TestCase):
    """Testing point generation functions."""

    def setUp(self):
        """Set up unittest."""
        file = open('./test/scenario', 'rb')
        self.s = pickle.load(file)
        file.close()
        self.random_generator = PointGen(self.s, random_point)
        # Set up a tournament with mock data
        point_selector = RandomSelector()
        for i in range(2):
            instance_selector = InstanceSet(self.s.instance_set, 1, 1)
            generated_points = [self.random_generator.point_generator(seed=42)
                                for _ in range(self.s.tournament_size *
                                               self.s.generator_multiple)]
            point_selector.select_points(
                generated_points,
                self.s.tournament_size, 0)
            instance_id, instances = instance_selector.get_subset(0)
            tourn = {}

            tourn_id = uuid.uuid4()
            gp = copy.deepcopy(generated_points)
            best_finisher = np.random.choice(gp)
            configuration_ids = []
            worst_finisher = np.random.choice(gp, size=5).tolist()
            for conf in generated_points:
                configuration_ids.append(conf.id)

            tourn[tourn_id] = \
                Tournament(tourn_id, [best_finisher], worst_finisher,
                           generated_points, configuration_ids, {},
                           ['instance_1.cnf'], 0)

            tourn[tourn_id].best_finisher = [tourn[tourn_id].configurations[0]]

            tourn[tourn_id].configurations.pop(i)
            tourn[tourn_id].worst_finisher = tourn[tourn_id].configurations
            tourn[tourn_id].configurations = []

        self.hist = tourn
        self.default_generator = PointGen(self.s, default_point)
        self.variable_graph_generator = PointGen(self.s, variable_graph_point)
        self.lhc_generator = PointGen(self.s, lhc_points)
        self.sm = SurrogateManager(self.s, seed=42)

    def test_default_point(self):
        """Testing default point generation."""
        conf = self.default_generator.point_generator()
        self.assertEqual(conf.conf, {'luby': False, 'rinc': 2.0,
                                     'cla-decay': 0.999,
                                     'phase-saving': 2,
                                     'strSseconds': '150',
                                     'bce-limit': 100000000,
                                     'param_1': -1})

    def test_random_point(self):
        """Testing random point generation."""
        conf = self.random_generator.point_generator(seed=42)
        self.assertEqual(conf.conf, {'luby': True,
                                     'rinc': 3.409974661894675,
                                     'cla-decay': 0.9175615966761705,
                                     'phase-saving': 1,
                                     'bce-limit': 9337277,
                                     'param_1': -1})

    def test_vg_point(self):
        """Testing variable graph point generation."""
        conf = self.variable_graph_generator.point_generator(
            mode=Mode.best_and_random,
            alldata=self.hist, lookback=0, seed=42)
        print('\nconf\n', conf)
        self.assertEqual(conf.conf, {'luby': True,
                                     'rinc': 3.409974661894675,
                                     'cla-decay': 0.9175615966761705,
                                     'phase-saving': 1,
                                     'bce-limit': 9337277,
                                     'param_1': -1})

    def test_lhc_point(self):
        """Testing variable graph point generation."""
        conf = self.lhc_generator.point_generator(
            n_samples=2, seed=42,
            lhs_type=LHSType.centered,
            criterion=Criterion.maximin)
        self.assertEqual(conf[0].conf, {'luby': True,
                                        'rinc': 3.275,
                                        'cla-decay': 0.9249975,
                                        'phase-saving': 0,
                                        'strSseconds': '250',
                                        'bce-limit': 50075000,
                                        'param_1': -2})

if __name__ == '__main__':
    unittest.main()
