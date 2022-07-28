"""This module contains simple tests for the point generation functions."""
import ray
import numpy as np
import uuid
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
from selector.surrogates.surrogates import SurrogateManager
from selector.pool import Status, Surrogates, Generator


def test_point_selection(scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    np.random.seed(42)

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

        results = ray.get(global_cache.get_results.remote())

        tourn, _ = MiniTournamentDispatcher().init_tournament(results,
                                                              points_to_run,
                                                              instances,
                                                              instance_id)

        tourn.configurations[i].generator = np.random.choice([*Generator])

        tourn.best_finisher = [tourn.configurations[i]]

        tourn.configurations.pop(i)
        tourn.worst_finisher = tourn.configurations

        tourn.configurations = []
        global_cache.put_tournament_history.remote(tourn)

    hist = ray.get(global_cache.get_tournament_history.remote())

    for tourn in hist.values():
        for conf in tourn.configuration_ids:
            global_cache.put_result.remote(uuid.UUID(str(conf)),
                                           instance_id,
                                           np.random.randint(2, 15))

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
            alldata=hist, lookback=i + 1, seed=(42 + i)))

    # print('\n Variable Graph configuration:\n\n', *var_conf, sep="\n\n")

    lhc_generator = PointGen(s, lhc_points, seed=42)

    lhc_conf = lhc_generator.point_generator(n_samples=5, seed=42,
                                             lhs_type=LHSType.centered,
                                             criterion=Criterion.maximin)

    # print('\n LHC configurations:\n\n', *lhc_conf, sep="\n\n")

    sm = SurrogateManager(s, seed=42)
    smac_conf = sm.suggest(Surrogates.SMAC, s, n_samples=5)

    confs = def_conf + ran_conf + var_conf + lhc_conf + smac_conf

    hps = HyperparameterizedSelector()

    configs_requested = 8
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weights = [weights for x in range(len(confs))]
    weights = np.array(weights)
    epoch = 4
    max_epochs = 256

    fg = FeatureGenerator()

    cutoff_time = s.cutoff_time
    results = ray.get(global_cache.get_results.remote())
    predicted_perf = []
    predicted_quals = []
    qap = False
    evaluated = []

    for epoch in range(10):
        features = fg.static_feature_gen(confs, epoch, max_epochs)
        features = np.concatenate((features,
                                   fg.diversity_feature_gen(confs, hist,
                                                            results,
                                                            cutoff_time,
                                                            s.parameter,
                                                            predicted_quals,
                                                            evaluated)),
                                  axis=1)

        for surrogate in sm.surrogates.keys():
            if sm.surrogates[surrogate].surr.model.rf is not None:
                predicted_perf = sm.predict(surrogate,
                                            confs,
                                            cutoff_time)

        features = np.concatenate((features,
                                  fg.dynamic_feature_gen(confs, hist,
                                                         predicted_perf,
                                                         sm, cutoff_time,
                                                         results)),
                                  axis=1)

        selected_ids = hps.select_points(s, confs, configs_requested, epoch,
                                         max_epochs, features, weights,
                                         results, max_evals=100, seed=42)

        for surrogate in sm.surrogates.keys():
            if sm.surrogates[surrogate].surr.model.rf is not None:
                if qap:
                    predicted_quals.extend(sm.predict(surrogate,
                                                      selected_ids,
                                                      cutoff_time))
                else:
                    predicted_quals.extend(sm.predict(surrogate,
                                                      evaluated,
                                                      cutoff_time))
                    qap = True

        evaluated.extend(selected_ids)

        for conf in selected_ids:
            global_cache.put_result.remote(conf.id,
                                           epoch,
                                           np.random.randint(2, 15))

        results = ray.get(global_cache.get_results.remote())

        for conf in selected_ids:
            for surrogate in sm.surrogates.keys():
                state = np.random.choice([Status.win, Status.cap,
                                          Status.timeout, Status.stop,
                                          Status.running])
                sm.update_surr(surrogate, results, conf, state, epoch)

        ran_conf = []
        for i in range(5):
            ran_conf.append(random_generator.point_generator(seed=42 + i))

        var_conf = []

        for i in range(5):
            var_conf.append(variable_graph_generator.point_generator(
                mode=Mode.random,
                alldata=hist, lookback=i + 1, seed=(42 + i)))

        lhc_conf = lhc_generator.point_generator(n_samples=5, seed=42,
                                                 lhs_type=LHSType.centered,
                                                 criterion=Criterion.maximin)

        smac_conf = sm.suggest(Surrogates.SMAC, s, n_samples=5)

        confs = ran_conf + var_conf + lhc_conf + smac_conf

        print('\n', selected_ids)
