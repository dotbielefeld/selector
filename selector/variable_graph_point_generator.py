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
from enum import Enum, IntEnum
from selector.pool import Configuration, ParamType
from selector.random_point_generator import random_set_conf
from selector.default_point_generator import check_conditionals, check_no_goods


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

    : param s: scenario object
    return: general variable graph structure
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

    : param config_label: current labeling of parameters
    : param C: first configuration
    : param N: second configuration
    : param cn: child node in the graph
    return: config_label
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

    : param config_label: current labeling of parameters
    : param C: first configuration
    : param N: second configuration
    return: config_label
    """
    if config_label[cn] == LabelType.C and cn not in C.conf:
        config_label.pop(cn, None)
    elif config_label[cn] == LabelType.N and cn not in N.conf:
        config_label.pop(cn, None)

    return config_label


def set_config_label(paths, config_label, cn, C, N):
    """
    Set label.

    : param paths: nodes visited until current node
    : param config_label: current labeling
    : param cn: current node
    : param C: first Configuration
    : param N: second configuration
    return: config_label
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
    Check if no goods violated and change value if so.

    : param s: scenario
    : param config_setting: parameter value setting
    : param label: label
    : param C: first Configuration
    : param N: second configuration
    return: config_setting
    """
    for ng in s.no_goods:
        param_1, param_2 = ng.keys()
        if ng[param_1] == config_setting[param_1]:
            if ng[param_2] == config_setting[param_2]:
                if label[param_1] == LabelType.C and param_1 in N.conf \
                        and N.conf[param_1] != ng[param_1]:
                    config_setting[param_1] = N.conf[param_1]
                elif label[param_2] == LabelType.C and param_2 in N.conf \
                        and N.conf[param_2] != ng[param_2]:
                    config_setting[param_2] = N.conf[param_2]
                elif label[param_1] == LabelType.N and param_1 in C.conf \
                        and C.conf[param_1] != ng[param_1]:
                    config_setting[param_1] = C.conf[param_1]
                elif label[param_2] == LabelType.N and param_2 in C.conf \
                        and C.conf[param_2] != ng[param_2]:
                    config_setting[param_2] = C.conf[param_2]
                else:
                    if random.uniform(0, 1) >= 0.5:
                        config_setting[param_1] = \
                            random_set_conf(s.parameter[param_1])
                    else:
                        config_setting[param_2] = \
                            random_set_conf(s.parameter[param_2])

    return config_setting


def graph_crossover(graph_structure, C, N, s):
    """
    Crossover according to variable graph.

    : param graph_structure: general variable graph structure
    : param C: configuration 1
    : param N: configuration 2
    : param s: scenario
    return: default configuration
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
    params_labeled = []

    while S:
        curr_node = S[0]
        S.pop(0)
        child_nodes = graph_structure[curr_node]
        for cn in child_nodes:
            if cn in paths:
                if curr_node not in paths[cn]:
                    paths[cn].append(cn)
                    config_label = set_config_label(paths, config_label,
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

    # Check conditionals and turn off parameters if violated
    cond_vio = check_conditionals(s, config_setting)
    for param in cond_vio:
        config_setting.pop(param, None)

    # Check no goods and reset values if violated
    ng_vio = check_no_goods(s, config_setting)
    while ng_vio:
        config_setting = reset_no_goods(s, config_setting,
                                        config_label, C, N)
        ng_vio = check_no_goods(s, config_setting)

    return config_setting


def choose_parents(mode, data, lookback):
    """
    Pick Configurations according to mode.

    : param mode: Enum, mode of parent selection
    : param data: Tournament data to select parents from
    : param lookback: int, how many tournaments from the past are included
    return: configurations C and N
    """
    if lookback < len(data):
        data = list(data.values())[len(data) - lookback:]
    else:
        data = list(data.values())

    if mode == Mode.random:
        conf_list = []

        for tourn in data:
            conf_list.extend(tourn.best_finisher)
            conf_list.extend(tourn.worst_finisher)

        C = np.random.choice(conf_list)
        C_ind = conf_list.index(C)
        conf_list.pop(C_ind)
        N = np.random.choice(conf_list)

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
            all_best.append(tourn.best_finisher[0])
            if len(tourn.best_finisher) > 1:
                all_worst.extend([*tourn.best_finisher[1:],
                                 *tourn.worst_finisher])
            else:
                all_worst.extend(tourn.worst_finisher)

        C = np.random.choice(all_best)

        N = np.random.choice(all_worst)

    return C, N


def variable_graph_point(s, identity, mode=Mode.random, data=False,
                         lookback=1, seed=False):
    """
    Configuration is generated via variable graph method.

    : param s: scenario object
    : param identity: uuid to identify configuration
    : param mode: Enum, mode of parent selection
    : param data: Tournament data to select parents from
    : param lookback: int, how many tournaments from the past are included
    : param seed: sets random seed
    return: configuration
    """
    if seed:
        np.random.seed(seed)

    if not data:
        print('No data given to variable point generator')
        exit()
    # Pick parent configurations
    C, N = choose_parents(mode, data, lookback)

    # Generate general graph structure
    graph_structure = variable_graph_structure(s)

    # Generate configuration via variable graph crossover
    config_setting = graph_crossover(graph_structure, C, N, s)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity, config_setting)

    return configuration
