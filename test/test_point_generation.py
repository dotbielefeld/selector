
"""This module contains simple tests for the point generation functions."""

from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point


def test_gen_funcs(s):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    random_generator = PointGen(s, random_point)

    # Set up a tournament with mock data
    global_cache = TargetAlgorithmObserver.remote()
    point_selector = RandomSelector()
    instance_selector = InstanceSet(s.instance_set, 1, 1)
    generated_points = [random_generator.point_generator()
                        for _ in range(s.tournament_size *
                                       s.generator_multiple)]
    points_to_run = point_selector.select_points(generated_points,
                                                 s.tournament_size, 0)
    instance_id, instances = instance_selector.get_subset(0)

    tourn, _ = MiniTournamentDispatcher().init_tournament(global_cache,
                                                          points_to_run,
                                                          instances,
                                                          instance_id)

    tourn.best_finisher = [1]

    default_generator = PointGen(s, default_point)
    param_1 = random_generator.point_generator()
    param_2 = default_generator.point_generator()

    print('\n Random configuration:\n\n', param_1, '\n')
    print('\n Default configuration:\n\n', param_2, '\n')

    variable_graph_generator = PointGen(s, variable_graph_point)

    param_3 = variable_graph_generator.point_generator(['best_and_random',
                                                        tourn])

    print('\n Variable Graph configuration:\n\n', param_3, '\n')
