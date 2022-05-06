"""This module contains the variable graph point generator."""

import pickle
import os
import random
import copy
import numpy as np
from selector.pool import Configuration, ParamType
from selector.random_point_generator import random_set_conf
from selector.default_point_generator import check_conditionals, check_no_goods
from os.path import exists


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

    for p in range(len(parameters) - 1):
        if parameters[p] not in graph_structure[parameters[p]]:
            graph_structure[parameters[p]] = []
        graph_structure[parameters[p]].append(parameters[p + 1])
        if s.parameter[p].condition:
            for cond in s.parameter[p].condition:
                if cond not in graph_structure[parameters[p]]:
                    graph_structure[parameters[p]].append(cond)

    for ng in s.no_goods:
        for param in ng:
            ng_params = list(ng.keys())
            for ngp in ng_params:
                if ngp not in graph_structure[param] and ngp != param:
                    graph_structure[param].append(ngp)

    return graph_structure


def decide_for_O(config_label, C, N, cn):
    """
    Decide new label in case of label O.

    : param config_label: current labeling of parameters
    : param C: first configuration
    : param N: second configuration
    return: config_label
    """
    if config_label[cn] == 'O':
        if cn in C.conf and cn in N.conf:
            if C.conf[cn] != N.conf[cn]:
                if random.uniform(0, 1) >= 0.5:
                    config_label[cn] = 'N'
                else:
                    config_label[cn] = 'C'
            else:
                config_label[cn] = 'C'
        else:
            if cn in C.conf and cn not in N.conf:
                if random.uniform(0, 1) >= 0.5:
                    config_label[cn] = 'C'
                else:
                    config_label.pop(cn, None)
            elif cn not in C.conf and cn in N.conf:
                if random.uniform(0, 1) >= 0.5:
                    config_label[cn] = 'N'
                else:
                    config_label.pop(cn, None)
            else:
                config_label.pop(cn, None)

    return config_label


def check_valid(config_label, C, N, cn):
    """
    Check if parameter is actually set in Configuration.

    : param config_label: current labeling of parameters
    : param C: first configuration
    : param N: second configuration
    return: config_label
    """
    if config_label[cn] == 'C' and cn not in C.conf:
        config_label.pop(cn, None)
    elif config_label[cn] == 'N' and cn not in N.conf:
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
        params = list(ng.keys())
        if ng[params[0]] == config_setting[params[0]]:
            if ng[params[1]] == config_setting[params[1]]:
                if label[params[0]] == 'C' and params[0] in N.conf and \
                        N.conf[params[0]] != ng[params[0]]:
                    config_setting[params[0]] = N.conf[params[0]]
                elif label[params[1]] == 'C' and params[1] in N.conf and \
                        N.conf[params[1]] != ng[params[1]]:
                    config_setting[params[1]] = N.conf[params[1]]
                elif label[params[0]] == 'N' and params[0] in C.conf and \
                        C.conf[params[0]] != ng[params[0]]:
                    config_setting[params[0]] = C.conf[params[0]]
                elif label[params[1]] == 'N' and params[1] in C.conf and \
                        C.conf[params[1]] != ng[params[1]]:
                    config_setting[params[1]] = C.conf[params[1]]
                else:
                    if random.uniform(0, 1) >= 0.5:
                        config_setting[params[0]] = \
                            random_set_conf(s.parameter[params[0]])
                    else:
                        config_setting[params[1]] = \
                            random_set_conf(s.parameter[params[1]])

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
        config_label[curr_node] = 'O'
    else:
        config_label[curr_node] = random.choice(['C', 'N'])

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
                    if config_label[cn] == 'N':
                        if cn in C.conf:
                            config_label[cn] = 'C'
                    else:
                        if cn in N.conf:
                            config_label[cn] = 'N'
                S.append(cn)

    for param, label in config_label.items():
        if label == 'N':
            config_setting[param] = N.conf[param]
        else:
            config_setting[param] = C.conf[param]

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


def choose_parents(CN):
    """
    Pick Configurations according to option set in CN[0].

    : param CN: list, contains ['option',Tournament]
    return: configurations C and N
    """
    if CN[0] == 'best_and_random':
        tournament = CN[1]
        if len(tournament.best_finisher) > 1:
            best_ind = np.random.choice(tournament.best_finisher)
        else:
            best_ind = tournament.best_finisher[0]

        conf_list = tournament.configurations

        C = conf_list.pop(best_ind)

        N = np.random.choice(conf_list)

    return C, N


def variable_graph_point(s, identity, CN):
    """
    Configuration is generated via variable graph method.

    : param s: scenario object
    : param identity: uuid to identify configuration
    return: default configuration
    """
    # Pick parent configurations 
    C, N = choose_parents(CN)

    # Generate general graph structure
    graph_structure = variable_graph_structure(s)

    # Generate configuration via variable graph crossover
    config_setting = graph_crossover(graph_structure, C, N, s)

    # Fill Configuration class with ID and parameter values
    configuration = Configuration(identity, config_setting)

    return configuration
