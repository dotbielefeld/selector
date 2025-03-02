"""This modulke contains functions to aggregate and discern performances."""
import numpy as np
import json
import os
import ray
import pickle
import math
import copy
from selector.log_setup import ConfEncoder
from selector.generators.default_point_generator import check_conditionals


def get_conf_time_out(results, configuration_id, instances_set):
    """
    Determine if a configuration timed out on any instance in a set

    Parameters
    ----------
    results : dic
        Dictionary of results: {conf_id: {instance: runtime}}
    configuration_id : uuid.UUID
        ID of Configuration.
    instances_set : list
        Set of instances

    Returns
    -------
    None or bool
        None if no results are present, True if conf timed out on any instance,
        False if it did not.
    """

    if configuration_id in list(results.keys()):
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]
        if np.isnan(conf_results_instances).any():
            return True
        else:
            return False
    else:
        return None


def get_censored_runtime_for_instance_set(results, configuration_id, instances_set):
    """
    For a configuration compute the total runtime needed only for instances in a set. 
    If there are no results for the conf, return 0. 
    Note that runs that were canceled by the monitor are not included since we count them as NaNs.

    Parameters
    ----------
    results : dict
        Dictionary of results: {conf_id: {instance: runtime}}
    configuration_id : uuid.UUID
        ID of the configuration
    instances_set : list
        List of instances

    Returns
    -------
    float
        Runtime of the configuration on instances
    """
    if configuration_id in results.keys():
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]
        runtime = np.nansum(list(conf_results_instances))

    return runtime


def get_runtime_for_instance_set_with_timeout(results, configuration_id, instances_set, timeout, par_penalty=1):
    """
    For a configuration compute the total runtime needed only for instances in a set. 
    If there are no results for the conf, return 0. 
    Note that runs that were canceled by the monitor are not included since we count them as NaNs.

    Parameters
    ----------
    results : dict
        Dictionary of results: {conf_id: {instance: runtime}}
    configuration_id : uuid.UUID
        ID of the configuration
    instances_set : list
        List of instances
    timeout : int
        Time limit for target algorithm runs in seconds.
    par_penalty : int
        PAR penalty for timeout.

    Returns
    -------
    float
        Runtime of the configuration on instances
    """
    if configuration_id in results.keys():
        conf_results_all_instances = results[configuration_id]
        conf_results_instances = [conf_results_all_instances[instance] for instance in
                                  conf_results_all_instances if instance in instances_set]

        return np.nansum(list(conf_results_instances)) + np.count_nonzero(np.isnan(list(conf_results_instances))) * (timeout * par_penalty)


def get_censored_runtime_of_configuration(results, configuration_id):
    """
    Get total runtime of a conf not conditioned on an instance set. Note that runs that were 
    canceled by the monitor are not included since we count them as nan's.

    Parameters
    ----------
    results : dict
        Results dictionary in the format {conf_id: {instance: runtime}}.
    configuration_id : int
        ID of the configuration.

    Returns
    -------
    float
        Total runtime of the configuration.
    """
    if configuration_id in results.keys():
        conf_results = results[configuration_id]
        runtime = np.nansum(list(conf_results.values()))
    return runtime


def get_instances_no_results(results, configuration_id, instance_set):
    """
    For a configuration get a list of instances we have no results for yet.

    Parameters
    ----------
    results : dict
        Dic of results: {conf_id: {instance: runtime}}.
    configuration_id : int
        Id of the configuration.
    instance_set : list
        List of instances.

    Returns
    -------
    list
        List of configurations the conf has not been run on.
    """
    not_run_on = copy.deepcopy(instance_set)

    if configuration_id in results.keys():
        configuration_results = results[configuration_id]

        instances_run_on = configuration_results.keys()

        for iro in instances_run_on:
            if iro in not_run_on:
                not_run_on.remove(iro)

    return not_run_on


def overall_best_update(tournaments, results, scenario, ac_runtime):
    """
    Over all tournaments get the best finisher with the most instance runs and shortest runtime and save that conf
    to a file.

    Parameters
    ----------
    cache : selector.ta_result_store.TargetAlgorithmObserver
            Tournament data cache.
    results : dict
        Dic of results: {conf_id: {instance: runtime}}.
    scenario : selector.scenario.Scenario
        AC scenario.
    ac_runtime : int
        Runtime spent by selector so far.
    """
    number_of_instances_run = {}
    runtime = {}
    confs = {}
    # For each tournament get the best finisher
    for t in tournaments:
        if t.best_finisher:
            best_winner = t.best_finisher[0]
            number_of_instances_run[best_winner.id] = len(results[best_winner.id])
            #runtime[best_winner.id] = get_censored_runtime_of_configuration(results, best_winner.id)
            runtime[best_winner.id] = get_runtime_for_instance_set_with_timeout(results, best_winner.id, t.instance_set, scenario.cutoff_time, scenario.par)
            confs[best_winner.id] = best_winner
    # If we have any best finisher we get those which ran on the most instances and then get the conf with the
    # shortest runtime on those
    if confs:
        max_number_instances = max(list(number_of_instances_run.values()))
        conf_with_max_number = [k for k, v in number_of_instances_run.items() if v == max_number_instances]

        conf_r_w_max_i = {k: runtime[k] for k in conf_with_max_number}
        conf_with_min_runtime = min(list(conf_r_w_max_i.values()))
        conf_with_min_runtime = [k for k, v in conf_r_w_max_i.items() if v == conf_with_min_runtime]

        clean_conf = copy.copy(confs[conf_with_min_runtime[0]].conf)
        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)

        with open(f"./selector/logs/{scenario.log_folder}/trajectory.json", 'a') as f:
            json.dump({str(confs[conf_with_min_runtime[0]].id): clean_conf, 'ac_runtime': ac_runtime}, f, cls=ConfEncoder)
            f.write(os.linesep)
