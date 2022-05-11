"""This module contains simple tests for the point generation functions."""
import ray
import unittest
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.scenario import Scenario, parse_args


class PointGenTest(unittest.TestCase):
    """Testing point generation functions."""

    def setUp(self):
        """Set up unittest."""
        parser = parse_args()
        self.s = Scenario("./test_data/test_scenario.txt", parser)
        self.random_generator = PointGen(self.s, random_point)
        # Set up a tournament with mock data
        global_cache = TargetAlgorithmObserver.remote()
        point_selector = RandomSelector()
        for i in range(2):
            instance_selector = InstanceSet(self.s.instance_set, 1, 1)
            generated_points = [self.random_generator.point_generator(seed=42)
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
        self.default_generator = PointGen(self.s, default_point)
        self.variable_graph_generator = PointGen(self.s, variable_graph_point)

    def test_default_point(self):
        """
        Testing default point generation.

        : param conf: generated default configuration
        """
        conf = self.default_generator.point_generator()
        self.assertEqual(conf.conf, {'luby': False, 'rinc': 2.0,
                                     'cla-decay': 0.999,
                                     'phase-saving': 2,
                                     'strSseconds': 150.0,
                                     'bce-limit': 100000000,
                                     'param_1': -1})

    def test_random_point(self):
        """
        Testing random point generation.

        : param conf: generated default configuration
        """
        conf = self.random_generator.point_generator(seed=42)
        self.assertEqual(conf.conf, {'luby': True,
                                     'rinc': 3.409974661894675,
                                     'cla-decay': 0.9175615966761705,
                                     'phase-saving': 1,
                                     'bce-limit': 9337277,
                                     'param_1': -1})

    def test_vg_point(self):
        """
        Testing variable graph point generation.

        : param conf: generated default configuration
        """
        conf = self.variable_graph_generator.point_generator(
            mode=Mode.best_and_random,
            data=self.hist, lookback=2, seed=42)
        self.assertEqual(conf.conf, {'luby': True,
                                     'rinc': 3.409974661894675,
                                     'cla-decay': 0.9175615966761705,
                                     'phase-saving': 1,
                                     'bce-limit': 9337277,
                                     'param_1': -1})

if __name__ == '__main__':
    unittest.main()
