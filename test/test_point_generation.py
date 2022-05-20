"""This module contains simple tests for the point generation functions."""
import ray
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion


def test_gen_funcs(scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    s = scenario("./test_data/test_scenario.txt", parser)

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

    print('\n Default configuration:\n\n', param_1, '\n')

    print('\n Random configuration:\n\n', param_2, '\n')

    variable_graph_generator = PointGen(s, variable_graph_point)

    param_3 = variable_graph_generator.point_generator(
        mode=Mode.best_and_random,
        data=hist, lookback=2, seed=42)

    print('\n Variable Graph configuration:\n\n', param_3, '\n')

    lhc_generator = PointGen(s, lhc_points)

    params_4 = lhc_generator.point_generator(n_samples=2, seed=42,
                                             lhs_type=LHSType.centered,
                                             criterion=Criterion.maximin)

    print('\n LHC configurations:\n\n', *params_4, sep="\n\n")
