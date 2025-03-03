u"""This module contains the variable graph point generator.

Based on the variable tree in GGA [Ansótegui, C., Sellmann, M., Tierney, K.:
A Gender-Based Genetic Algorithm for the Automatic Configuration of Algorithms.
In: Principles and Practice of Constraint Programming - CP 2009.
pp. 142–157 (09 2009).]
"""

import pickle
import os
import random
import copy
import numpy as np
import itertools
import math
from enum import Enum, IntEnum
from selector.pool import Configuration, ParamType, Generator
from selector.generators.random_point_generator import (
    random_set_conf,
    reset_conditionals
)
from selector.generators.default_point_generator import (
    check_conditionals,
    check_no_goods
)


class LabelType(IntEnum):
    """Contains the types of labels a node can assume."""

    C = 1
    N = 2
    O = 3


class Mode(Enum):
    """Contains the types of modes for choosing parent configurations."""

    random = 1
    only_best = 2
    best_and_random = 3


def variable_graph_structure(s):
    """
    General variable graph structure is read from scenario.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.

    Returns
    -------
    dict of lists
        General variable graph structure.
    """
    parameters = [param.name for param in s.parameter]
    graph_structure = {}
    for p in parameters:
        graph_structure[p] = []

    for pi, pj in zip(s.parameter[:-1], s.parameter[1:]):
        if pi.name not in graph_structure[pi.name]:
            graph_structure[pi.name] = []
        graph_structure[pi.name].append(pj.name)
        if pi.condition:
            for cond in pi.condition:
                if cond not in graph_structure[pi.name]:
                    graph_structure[pi.name].append(cond)

    for ng in s.no_goods:
        for param in ng:
            for ngp, _ in ng.items():
                if ngp not in graph_structure[param] and ngp != param:
                    graph_structure[param].append(ngp)

    return graph_structure


def decide_for_O(config_label, C, N, cn):
    """
    Decide new label in case of label O.

    Parameters
    ----------
    config_label : dict
        Current labeling of parameters.
    C : dict
        First configuration.
    N : dict
        Second configuration.
    cn : str
        Child node in the graph.

    Returns
    -------
    dict
        Updated configuration label.
    """
    if config_label[cn] == LabelType.O:
        if cn in C.conf and cn in N.conf:
            config_label[cn] = LabelType.C
            if C.conf[cn] != N.conf[cn] and random.uniform(0, 1) >= 0.5:
                config_label[cn] = LabelType.N
        elif cn in C.conf and cn not in N.conf:
            if random.uniform(0, 1) >= 0.5:
                config_label[cn] = LabelType.C
            else:
                del config_label[cn]
        elif cn not in C.conf and cn in N.conf:
            if random.uniform(0, 1) >= 0.5:
                config_label[cn] = LabelType.N
            else:
                del config_label[cn]
        else:
            del config_label[cn]

    return config_label


def check_valid(config_label, C, N, cn):
    """
    Check if parameter is actually set in Configuration.

    Parameters
    ----------
    config_label : dict
        Current labeling of parameters.
    C : dict
        First configuration.
    N : dict
        Second configuration.

    Returns
    -------
    dict
        Updated configuration label.
    """
    if cn in config_label:
        if config_label[cn] == LabelType.C and cn not in C.conf:
            config_label.pop(cn, None)
        elif config_label[cn] == LabelType.N and cn not in N.conf:
            config_label.pop(cn, None)

    return config_label


def set_config_label(paths, config_label, cn, C, N):
    """
    Set label.

    Parameters
    ----------
    paths : dict
        Nodes visited until the current node.
    config_label : selector.generators.variable_graph_point_generator.LabelType
        Current labeling.
    cn : str
        Current node.
    C : dict
        First configuration.
    N : dict
        Second configuration.

    Returns
    -------
    dict
        Updated config_label.
    """
    parent_nodes = copy.copy(paths[cn])
    parent_nodes.remove(cn)
    parent_labels = [config_label[x] for x in parent_nodes
                     if x in config_label]
    values, counts = np.unique(parent_labels,
                               return_counts=True)
    config_label[cn] = random.choices(values, weights=counts,
                                      k=1)[0]
    config_label = decide_for_O(config_label, C, N, cn)
    config_label = check_valid(config_label, C, N, cn)

    return config_label


def reset_no_goods(s, config_setting, label, C, N):
    """
    Check if no goods are violated and change value if so.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    config_setting : dict
        Parameter value setting.
    label : str
        Label.
    C : dict
        First configuration.
    N : dict
        Second configuration.

    Returns
    -------
    dict
        Updated config_setting.
    """
    for ng in s.no_goods:
        params = list(ng.keys())
        if all(ng[p] == config_setting[p] for p in params):
            for p in params:
                if label[p] == LabelType.C and p in N.conf \
                        and N.conf[p] != ng[p]:
                    config_setting[p] = N.conf[p]
                elif label[p] == LabelType.N and p in C.conf \
                        and C.conf[p] != ng[p]:
                    config_setting[p] = C.conf[p]
                else:
                    for param in s.parameter:
                        if p == param.name:
                            config_setting[p] = \
                                random_set_conf([param])[p]

    return config_setting


def graph_crossover(graph_structure, C, N, s):
    """
    Crossover according to variable graph.

    Parameters
    ----------
    graph_structure : dict of list
        General variable graph structure.
    C : dict
        Configuration 1.
    N : dict
        Configuration 2.
    s : selector.scenario.Scenario
        AC scenario.

    Returns
    -------
    dict
        New configuration setting.
    """
    params = list(graph_structure.keys())
    curr_node = params[0]

    config_label = {}
    config_setting = {}
    paths = {}
    S = [curr_node]

    # Label root node
    if C.conf[curr_node] == N.conf[curr_node] or\
            len(graph_structure[curr_node]) > 1:
        config_label[curr_node] = LabelType.O
    else:
        config_label[curr_node] = random.choice([LabelType.C, LabelType.N])

    paths[curr_node] = [curr_node]

    while S:
        curr_node = S[0]
        S.pop(0)
        child_nodes = graph_structure[curr_node]
        for cn in child_nodes:
            if curr_node != cn:
                if cn in paths:
                    if cn not in paths[cn]:
                        paths[cn].append(cn)
                        config_label = set_config_label(paths,
                                                        config_label,
                                                        cn, C, N)
                        S.append(cn)
                else:
                    paths[cn] = [*paths[curr_node], cn]
                    config_label = set_config_label(paths, config_label,
                                                    cn, C, N)
                    S.append(cn)

                if random.uniform(0, 1) < 0.1:
                    if cn in config_label:
                        if config_label[cn] == LabelType.N and cn in C.conf:
                            config_label[cn] = LabelType.C
                        elif config_label[cn] == LabelType.C and cn in N.conf:
                            config_label[cn] = LabelType.N
                    S.append(cn)

    for param, label in config_label.items():
        config_setting[param] = N.conf[param] if label == LabelType.N \
            else C.conf[param]

    param_info = {}
    for param in s.parameter:
        param_info[param.name] = {'type': param.type, 'bound': param.bound}

    for param, value in config_setting.items():
        if param_info[param]['type'] == ParamType.categorical:
            if random.uniform(0, 1) < 0.1:
                config_setting[param] = \
                    random.choice(param_info[param]['bound'])
        else:
            if random.uniform(0, 1) < 0.1:
                mutation = param_info[param]['bound'][0] - 1
                while mutation < param_info[param]['bound'][0] or \
                        mutation > param_info[param]['bound'][1]:
                    mu = config_setting[param]
                    sigma = (param_info[param]['bound'][1] -
                             param_info[param]['bound'][0]) * 0.1
                    mutation = \
                        np.random.normal(mu, sigma, 1)[0]
                if param_info[param]['type'] == ParamType.continuous:
                    config_setting[param] = mutation
                else:
                    config_setting[param] = round(mutation)

    # Check no goods and reset values if violated
    ng_vio = check_no_goods(s, config_setting)
    while ng_vio:
        config_setting = reset_no_goods(s, config_setting,
                                        config_label, C, N)
        ng_vio = check_no_goods(s, config_setting)

    return config_setting


def choose_parents(mode, data, lookback, results):
    """
    Pick configurations according to mode.

    Parameters
    ----------
    mode : selector.generators.variable_graph_point_generator.Mode
        Mode of parent selection.
    data : dict of selector.pool.Tournament
        Tournament data to select parents from.
    lookback : int
        Number of past tournaments included.

    Returns
    -------
    tuple
        - **C** : dict,
          Configuration C.
        - **N** : dict,
          Configuration N.
    """
    if lookback < len(data):
        data = list(data.values())[:len(data) - lookback]
    else:
        data = list(data.values())

    if mode == Mode.random:
        conf_list = []

        for tourn in data:
            conf_list.extend(tourn.best_finisher)
            conf_list.extend(tourn.worst_finisher)

        if conf_list:
            C = np.random.choice(conf_list)
            C_ind = conf_list.index(C)
            conf_list.pop(C_ind)
        else:
            C = np.random.choice(tourn.configurations)

        if conf_list:
            N = np.random.choice(conf_list)
        else:
            N = np.random.choice(tourn.configurations)

    elif mode == Mode.only_best:
        all_best = []
        for tourn in data:
            all_best.append(tourn.best_finisher[0])

        if len(all_best) > 1:
            C = np.random.choice(all_best)
            best_ind = all_best.index(C)
            all_best.pop(best_ind)
            N = np.random.choice(all_best)

        else:
            mode = Mode.best_and_random

    elif mode == Mode.best_and_random:
        all_best = []
        all_worst = []
        for tourn in data:
            all_best.extend(tourn.best_finisher)

        performance = []
        for ab in all_best:
            perf = 0
            for res in results[ab.id].values():
                perf += res
            perf = perf / len(results[ab.id])
            performance.append(perf)

        take_best = math.ceil(len(all_best) * 0.1)
        sort_index = np.argsort(performance)
        the_best = sort_index[:take_best]
        best = []
        for tb in the_best:
            best.append(all_best[tb])

        all_best = best
        for tourn in data:
            for tbf in tourn.best_finisher:
                if tbf in all_best:
                    continue
                else:
                    all_worst.append(tbf)
            all_worst.extend(tourn.worst_finisher)

        all_configs = []

        if all_best:
            C = np.random.choice(all_best)
        else:
            for tourn in data:
                all_configs.extend(tourn.configurations)
            C = np.random.choice(all_configs)

        if all_worst:
            N = np.random.choice(all_worst)
        else:
            if all_configs:
                pass
            else:
                for tourn in data:
                    all_configs.extend(tourn.configurations)
            N = None
            while N != C:
                N = np.random.choice(all_configs)

    return C, N


def variable_graph_point(s, identity, results, mode=Mode.best_and_random,
                         alldata=False, lookback=1, seed=False):
    """
    Configuration is generated via variable graph method.

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    identity : uuid.UUID
        UUID to identify configuration.
    mode : selector.generators.variable_graph_point_generator.Mode
        Mode of parent selection.
    data : dict of selector.pool.Tournament
        Tournament data to select parents from.
    lookback : int
        Number of past tournaments included.
    seed : int
        Random seed.

    Returns
    -------
    selector.pool.Configuration
        Configuration generated with GGA graph.
    """
    if seed:
        np.random.seed(seed)

    if not alldata:
        print('No data given to variable point generator')
        exit()
    # Pick parent configurations
    data = copy.copy(alldata)
    C, N = choose_parents(mode, data, lookback, results)

    # Generate general graph structure
    graph_structure = variable_graph_structure(s)

    # Generate configuration via variable graph crossover
    config_setting = graph_crossover(graph_structure, C, N, s)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity,
                                  config_setting,
                                  Generator.var_graph)

    return configuration
