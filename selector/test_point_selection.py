"""This module contains simple tests for the point generation functions."""
import ray
import numpy as np
import sys
import os
import time
import cProfile
sys.path.append(os.getcwd())
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import default_point
from selector.generators.variable_graph_point_generator import variable_graph_point, Mode
from selector.generators.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator
from selector.generators.surrogates.surrogates import SurrogateManager
from selector.pool import Surrogates, Tournament


def test_point_selection(scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    s = scenario("cadical/scenario_fuzz.txt", parser)

    evaluated = []
    predicted_quals = []
    sm = SurrogateManager(s)

    random_generator = PointGen(s, random_point, seed=42)

    # Set up a tournament with mock data
    global_cache = TargetAlgorithmObserver.remote(s)
    point_selector = RandomSelector()
    for i in range(5):
        instance_selector = InstanceSet(s.instance_set, 2, 1)
        generated_points = [random_generator.point_generator(seed=42 + i)
                            for _ in range(s.tournament_size *
                                           s.generator_multiple)]
        points_to_run = point_selector.select_points(generated_points,
                                                     s.tournament_size, 0)
        instance_id, instances = instance_selector.get_subset(0)

        re = ray.get(global_cache.get_results.remote())

        tourn, _ = MiniTournamentDispatcher().init_tournament(re,
                                                              points_to_run,
                                                              instances,
                                                              instance_id)

        tourn.best_finisher = [tourn.configurations[i]]

        tourn.configurations.pop(i)
        tourn.worst_finisher = tourn.configurations
        tourn.configurations = []
        global_cache.put_tournament_history.remote(tourn)

    hist = ray.get(global_cache.get_tournament_history.remote())

    re = ray.get(global_cache.get_results.remote())
    re = ray.get(global_cache.get_results.remote())

    default_generator = PointGen(s, default_point, seed=42)

    def_conf = [default_generator.point_generator()]

    ran_conf = []
    for i in range(5):
        ran_conf.append(random_generator.point_generator(seed=42 + i))

    # print('\n Default configuration:\n\n', def_conf, '\n')

    # print('\n Random configuration:\n\n', *ran_conf, sep="\n\n")

    variable_graph_generator = PointGen(s, variable_graph_point, seed=42)

    var_conf = []

    time_start = time.time()

    for i in range(4):
        
        var_conf.append(variable_graph_generator.point_generator(
            results=re,
            mode=Mode.random,
            alldata=hist, lookback=i + 1, seed=(42 + i)))
    print('\nVar Time:', time.time() - time_start)

    # print('\n Variable Graph configuration:\n\n', *var_conf, sep="\n\n")

    lhc_generator = PointGen(s, lhc_points, seed=42)

    for i in range(5):

        time_start = time.time()

        lhc_conf = lhc_generator.point_generator(n_samples=6, seed=42,
                                                 lhs_type=LHSType.centered,
                                                 criterion=Criterion.maximin)

        print('\nLHC Time:', time.time() - time_start)

    # print('\n LHC configurations:\n\n', *lhc_conf, sep="\n\n")

    import random

    re = {}

    for i in range(4):

        ins_id, ins = instance_selector.get_subset(i + 1)

        instances += ins

        time_start = time.time()

        sugg, _ = sm.suggest(Surrogates.CPPL, s, 6, hist, re,
                             instances)

        print(i, '\nCPPL Gen Time:', time.time() - time_start)

        ids = []

        for c in sugg:            
            for inst in ins:
                ids.append(c.id)
                if c.id in re:
                    re[c.id][inst] = random.randint(1, 10)
                else:
                    re[c.id] = {inst: random.randint(1, 10)}

        re_tourn = Tournament(id=i, best_finisher=[random.choice(sugg)],
                              worst_finisher=[],
                              configurations=[sugg], configuration_ids=ids,
                              ray_object_store={'a': 1}, instance_set=ins,
                              instance_set_id=ins_id)

        time_start = time.time()

        sm.update_surr(Surrogates.CPPL, re_tourn, sugg, re, [], 10 + i)

        print(i, '\nCPPL Update Time:', time.time() - time_start)

    confs = def_conf + ran_conf + var_conf + lhc_conf
    generated_points = confs

    for tornam in hist.values():
        for c in tornam.configuration_ids:
            re[c] = {'1': 3, '2': 5, '3': 7}

    hps = HyperparameterizedSelector()

    configs_requested = 8
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weights = [weights for x in range(len(confs))]
    weights = np.array(weights)
    epoch = 4
    max_epochs = 256

    fg = FeatureGenerator()

    evaluated.extend(generated_points)

    print('Start')

    for h in hist.values():
        # print(h)
        for c in h.best_finisher + h.worst_finisher:
            print(c.generator)

    '''

    for epoch in range(3):
        features = \
            fg.static_feature_gen(generated_points, epoch, max_epochs)

        #print('1', features, len(features[0]))
        features = np.concatenate(
            (features, fg.diversity_feature_gen(generated_points, hist,
                                                re, 30,
                                                s.parameter,
                                                predicted_quals,
                                                evaluated)),
            axis=1)
        #print('2', features, len(features[0]))

        features = np.concatenate((features,
                                   fg.dynamic_feature_gen(generated_points,
                                                          hist,
                                                          None,
                                                          sm, 30,
                                                          re,
                                                          instances)),
                                  axis=1)
        #print('3', features, len(features[0]))

        selected_ids = hps.select_points(s, confs, configs_requested, epoch,
                                         max_epochs, features, weights, re,
                                         max_evals=100, seed=42)
        print('\n\n')
    '''
