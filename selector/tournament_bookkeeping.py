"""This module contains functions for bookkeeping of tournaments."""
from selector.ta_execution import tae_from_cmd_wrapper_rt, tae_from_cmd_wrapper_quality
import time


def get_tournament_membership(tournaments, conf):
    """
    For a list of tournaments, determine which ones a configuration is a member of.

    Parameters
    ----------
    tournaments : list
        List of tournaments.
    conf : object
        Configuration.

    Returns
    -------
    list
        Tournaments the configuration is a member of.
    """
    for t in tournaments:
        if conf.id in t.configuration_ids or conf.id in t.worst_finisher or conf.id in t.best_finisher:
            return t


def get_get_tournament_membership_with_ray_id(task_id, tournaments):
    """
    For a Ray task ID, return the tournament it belongs to.

    Parameters
    ----------
    task_id : object
        The Ray task ID.
    tournaments : list
        List of tournaments.

    Returns
    -------
    selector.pool.Tournament
        The tournament the task ID belongs to.
    """
    ob_t = None
    for t in tournaments:
        t_objects = t.ray_object_store
        for confs, instance_objects in t_objects.items():
            for inst, ob in instance_objects.items():
                if ob == task_id:
                    ob_t = t
                    pass
    return ob_t


def get_tasks(taskdic, tasks):
    """
    Map back a Ray object to the configuration/instance pair.

    Parameters
    ----------
    taskdic : dict
        Nested dictionary of the form `{conf: {instance: ray object}}`.
    tasks : list
        List of Ray objects that are currently running.

    Returns
    -------
    list
        List of `[conf, instance]` pairs that are currently running.
    """
    running_tasks = []
    for conf, instance in taskdic.items():
        for instance_name, object in instance.items():
            if object in tasks:
                running_tasks.append([conf, instance_name])
    return running_tasks


def update_tasks(tasks, next_task, tournament, global_cache, ta_wrapper, scenario):
    """
    Update tasks and add new tasks if needed.

    Parameters
    ----------
    tasks : list
        List of Ray objects.
    next_task : list
        List of `[conf, instance]` pairs.
    tournament : object
        Tournament the next task is part of.
    global_cache : object
        Ray cache.
    ta_wrapper : object
        Target algorithm wrapper.
    scenario : object
        Scenario configuration.

    Returns
    -------
    list
        Updated list of Ray objects.
    """
    for t in next_task:
        if t[1] is not None:
            # TODO need to change the wrapper to something more generic here
            if scenario.run_obj == "runtime":
                task = tae_from_cmd_wrapper_rt.remote(t[0], t[1], global_cache, ta_wrapper, scenario)
            elif scenario.run_obj == "quality":
                task = tae_from_cmd_wrapper_quality.remote(t[0], t[1], global_cache, ta_wrapper, scenario)
            tasks.append(task)
            # We also add the ray object id to the tournament to latter map the id back
            if t[0].id not in tournament.ray_object_store.keys():
                tournament.ray_object_store[t[0].id] = {t[1]: task}
            else:
                tournament.ray_object_store[t[0].id][t[1]] = task
    return tasks


def termination_check(termination_criterion, main_loop_start, total_runtime, total_tournament_number,
                      tournament_counter):
    """
    Check the termination criterion for the main tournament loop and return `True` 
    if the criterion is not met yet.

    Parameters
    ----------
    termination_criterion : str
        Termination criterion for the tournament main loop.
    main_loop_start : int
        Time of the start of the tournament main loop.
    total_runtime : int, optional
        Total runtime for the main loop when the termination criterion is "total_runtime".
    total_tournament_number : int, optional
        Total number of tournaments for the main loop when the termination criterion is 
        "total_tournament_number".
    tournament_counter : int
        Number of tournaments that have already finished.

    Returns
    -------
    bool
        `True` if the termination criterion is not met, `False` otherwise.
    """
    if termination_criterion == "total_runtime":
        return time.time() - main_loop_start < total_runtime

    elif termination_criterion == "total_tournament_number":
        return tournament_counter < total_tournament_number

    else:
        return time.time() - main_loop_start < total_runtime
