"""This module contains simple tests for the point generation functions."""
import ray
import unittest
import argparse
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points


class Point_Gen_Test(unittest.TestCase):
    """Testing point generation functions."""

    def test_default_point(self, conf):
        """
        Testing default point generation.

        : param conf: generated default configuration
        """
        self.assertEqual(conf, {'luby': False, 'rinc': 2.0,
                                'cla-decay': 0.999,
                                'phase-saving': 2,
                                'strSseconds': 150.0,
                                'bce-limit': 100000000,
                                'param_1': -1})

    def test_random_point(self, conf):
        """
        Testing random point generation.

        : param conf: generated default configuration
        """
        self.assertEqual(conf, {'luby': True,
                                'rinc': 3.409974661894675,
                                'cla-decay': 0.9175615966761705,
                                'phase-saving': 1,
                                'bce-limit': 9337277,
                                'param_1': -1})

    def test_vg_point(self, conf):
        """
        Testing variable graph point generation.

        : param conf: generated default configuration
        """
        self.assertEqual(conf, {'luby': True,
                                'rinc': 3.409974661894675,
                                'cla-decay': 0.9175615966761705,
                                'phase-saving': 1,
                                'bce-limit': 9337277,
                                'param_1': -1})

    def test_lhc_point(self):
        """
        Testing variable graph point generation.

        : param conf: generated default configuration
        """
        self.assertEqual(conf, {})


def test_gen_funcs(Scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    s = Scenario("./test_data/test_scenario.txt", parser)

    random_generator = PointGen(s, random_point)

    # Set up a tournament with mock data
    global_cache = TargetAlgorithmObserver.remote()
    point_selector = RandomSelector()
    for i in range(2):
        instance_selector = InstanceSet(s.instance_set, 1, 1)
        generated_points = [random_generator.point_generator(seed=42)
                            for _ in range(s.tournament_size *
                                           s.generator_multiple)]
        points_to_run = point_selector.select_points(generated_points,
                                                     s.tournament_size, 0)
        instance_id, instances = instance_selector.get_subset(0)

        tourn, _ = MiniTournamentDispatcher().init_tournament(global_cache,
                                                              points_to_run,
                                                              instances,
                                                              instance_id)

        tourn.best_finisher = [tourn.configurations[i]]

        tourn.configurations.pop(i)
        tourn.worst_finisher = tourn.configurations
        tourn.configurations = []
        global_cache.put_tournament_history.remote(tourn)

    hist = ray.get(global_cache.get_tournament_history.remote())

    default_generator = PointGen(s, default_point)

    param_1 = default_generator.point_generator()
    param_2 = random_generator.point_generator(seed=42)

    Test = Point_Gen_Test()

    Test.test_default_point(param_1.conf)

    Test.test_random_point(param_2.conf)

    print('\n Default configuration:\n\n', param_1, '\n')

    print('\n Random configuration:\n\n', param_2, '\n')

    variable_graph_generator = PointGen(s, variable_graph_point)

    param_3 = variable_graph_generator.point_generator(
        mode=Mode.best_and_random,
        data=hist, lookback=2, seed=42)

    Test.test_vg_point(param_3.conf)

    print('\n Variable Graph configuration:\n\n', param_3, '\n')

    lhc_generator = PointGen(s, lhc_points)

    params_4 = lhc_generator.point_generator(n_samples=2)

    print('\n LHC configurations:\n\n', *params_4, sep="\n\n")
