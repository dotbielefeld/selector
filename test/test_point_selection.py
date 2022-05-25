"""This module contains simple tests for the point generation functions."""
import ray
import numpy as np
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator


def test_point_selection(scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    s = scenario("./test_data/test_scenario.txt", parser)

    random_generator = PointGen(s, random_point, seed=42)

    # Set up a tournament with mock data
    global_cache = TargetAlgorithmObserver.remote()
    point_selector = RandomSelector()
    for i in range(5):
        instance_selector = InstanceSet(s.instance_set, 1, 1)
        generated_points = [random_generator.point_generator(seed=42 + i)
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

    default_generator = PointGen(s, default_point, seed=42)

    def_conf = [default_generator.point_generator()]

    ran_conf = []
    for i in range(5):
        ran_conf.append(random_generator.point_generator(seed=42 + i))

    # print('\n Default configuration:\n\n', def_conf, '\n')

    # print('\n Random configuration:\n\n', *ran_conf, sep="\n\n")

    variable_graph_generator = PointGen(s, variable_graph_point, seed=42)

    var_conf = []

    for i in range(4):
        var_conf.append(variable_graph_generator.point_generator(
            mode=Mode.random,
            data=hist, lookback=i + 1, seed=(42 + i)))

    # print('\n Variable Graph configuration:\n\n', *var_conf, sep="\n\n")

    lhc_generator = PointGen(s, lhc_points, seed=42)

    lhc_conf = lhc_generator.point_generator(n_samples=5, seed=42,
                                             lhs_type=LHSType.centered,
                                             criterion=Criterion.maximin)

    # print('\n LHC configurations:\n\n', *lhc_conf, sep="\n\n")

    confs = def_conf + ran_conf + var_conf + lhc_conf

    hps = HyperparameterizedSelector()

    configs_requested = 8
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weights = [weights for x in range(len(confs))]
    weights = np.array(weights)
    epoch = 4
    max_epochs = 256

    fg = FeatureGenerator()
    features = fg.static_feature_gen(confs, epoch, max_epochs)

    for epoch in range(3):
        selected_ids = hps.select_points(s, confs, configs_requested, epoch,
                                         max_epochs, features, weights,
                                         max_evals=100, seed=42)
        print(selected_ids)
        print(len(selected_ids))
