"""This module contains simple tests for the point generation functions."""
import ray
import numpy as np
import uuid
from selector.ta_result_store import TargetAlgorithmObserver
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.pointselector import RandomSelector, HyperparameterizedSelector
from selector.instance_sets import InstanceSet
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import default_point
from selector.generators.variable_graph_point_generator import \
    variable_graph_point, Mode
from selector.generators.lhs_point_generator import lhc_points, LHSType, \
    Criterion
from selector.selection_features import FeatureGenerator
from selector.generators.surrogates.surrogates import SurrogateManager
from selector.pool import Status, Surrogates, Generator


def test_point_selection(scenario, parser):
    """
    Testing point generation functions and printing configurations.

    : param scenario: scenario object
    """
    np.random.seed(42)

    s = scenario
    random_generator = PointGen(s, random_point, seed=42)

    # Set up a tournament with mock data
    global_cache = TargetAlgorithmObserver.remote(scenario)
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

    result_tournament = hist[list(hist.keys())[4]]

    for tourn in hist.values():
        for conf in tourn.configuration_ids:
            global_cache.put_result.remote(uuid.UUID(str(conf)),
                                           result_tournament.instance_set[0],
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
            results=results, mode=Mode.random,
            alldata=hist, lookback=i + 1, seed=(42 + i)))

    # print('\n Variable Graph configuration:\n\n', *var_conf, sep="\n\n")

    lhc_generator = PointGen(s, lhc_points, seed=42)

    lhc_conf = lhc_generator.point_generator(n_samples=5, seed=42,
                                             lhs_type=LHSType.centered,
                                             criterion=Criterion.maximin)

    # print('\n LHC configurations:\n\n', *lhc_conf, sep="\n\n")

    results = ray.get(global_cache.get_results.remote())

    sm = SurrogateManager(s, seed=42)
    smac_conf = sm.suggest(Surrogates.SMAC, s, 5, _, _, _)
    ggapp_conf = sm.suggest(Surrogates.GGApp, s, 5, hist, results, _)
    cppl_conf = sm.suggest(Surrogates.CPPL, s, 5, _, _, s.instance_set)[0]

    confs = \
        def_conf + ran_conf + var_conf + lhc_conf + smac_conf + ggapp_conf \
        + cppl_conf

    hps = HyperparameterizedSelector()

    configs_requested = 8
    epoch = 4
    max_epochs = 256

    fg = FeatureGenerator()

    cutoff_time = s.cutoff_time
    results = ray.get(global_cache.get_results.remote())
    predicted_perf = []
    predicted_quals = []
    qap = False
    evaluated = []

    results = ray.get(global_cache.get_results.remote())

    for epoch in range(5):
        result_tournament = hist[list(hist.keys())[4]]

        all_configs \
            = result_tournament.best_finisher \
            + result_tournament.worst_finisher

        terminations = []

        # print('\nall_configs\n', all_configs, '\n')

        for surrogate in sm.surrogates.keys():
            sm.update_surr(surrogate, result_tournament, all_configs,
                           results, terminations)

        features = fg.static_feature_gen(confs, epoch, max_epochs)
        features = np.concatenate((features,
                                   fg.diversity_feature_gen(confs, hist,
                                                            results,
                                                            cutoff_time,
                                                            s.parameter,
                                                            predicted_quals,
                                                            evaluated)),
                                  axis=1)
        '''
        predicted_perf = []

        for surrogate in sm.surrogates.keys():
            if surrogate is Surrogates.SMAC:
                if sm.surrogates[surrogate].surr.model.rf is not None:
                    predicted_perf.append(sm.predict(surrogate,
                                                     confs,
                                                     cutoff_time,
                                                     _))
            else:
                predicted_perf.append(sm.predict(surrogate,
                                                 confs,
                                                 cutoff_time,
                                                 s.instance_set))
        '''

        features = np.concatenate((features,
                                  fg.dynamic_feature_gen(confs, hist,
                                                         _,
                                                         sm, cutoff_time,
                                                         results,
                                                         s.instance_set)),
                                  axis=1)

        print(len(features[0]))

        set_weights = [value for hp, value in s.__dict__.items()
                       if hp[:2] == 'w_']
        weights = [set_weights for x in range(len(confs))]
        weights = np.array(weights)

        selected_ids = hps.select_points(s, confs, configs_requested, epoch,
                                         max_epochs, features, weights,
                                         results, max_evals=100, seed=42)

        for surrogate in sm.surrogates.keys():
            if surrogate is Surrogates.SMAC:
                if sm.surrogates[surrogate].surr.model.rf is not None:
                    if qap:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          selected_ids,
                                                          cutoff_time,
                                                          _))
                    else:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          evaluated,
                                                          cutoff_time,
                                                          _))
                        qap = True
            else:
                if qap:
                    predicted_quals.extend(sm.predict(surrogate,
                                                      selected_ids,
                                                      cutoff_time,
                                                      s.instance_set))
                else:
                    predicted_quals.extend(sm.predict(surrogate,
                                                      evaluated,
                                                      cutoff_time,
                                                      s.instance_set))
                    qap = True

        evaluated.extend(selected_ids)

        for conf in selected_ids:
            global_cache.put_result.remote(conf.id,
                                           result_tournament.instance_set[0],
                                           np.random.randint(2, 15))

        results = ray.get(global_cache.get_results.remote())

        ran_conf = []
        for i in range(5):
            ran_conf.append(random_generator.point_generator(seed=42 + i))

        var_conf = []

        for i in range(5):
            var_conf.append(variable_graph_generator.point_generator(
                results=results, mode=Mode.best_and_random,
                alldata=hist, lookback=i + 1, seed=(42 + i)))

        lhc_conf = lhc_generator.point_generator(n_samples=5, seed=42,
                                                 lhs_type=LHSType.centered,
                                                 criterion=Criterion.maximin)

        smac_conf = sm.suggest(Surrogates.SMAC, s, 5, _, _, _)
        ggapp_conf = sm.suggest(Surrogates.GGApp, s, 5, hist, results, _)
        cppl_conf = sm.suggest(Surrogates.CPPL, s, 5, _, _, s.instance_set)[0]

        confs = ran_conf + var_conf + lhc_conf + smac_conf + ggapp_conf + \
            cppl_conf

        # print('\n', selected_ids)
