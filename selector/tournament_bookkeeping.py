import os
from ta_execution import tae_from_cmd_wrapper

def get_tournament_membership(tournaments, conf):
    """
    For a list of tournaments, determine of which a conf is a member.
    :param tournaments: List
    :param conf: Conf
    :return:
    """
    for t in tournaments:
        if conf.id in t.configuration_ids or conf.id in t.worst_finisher or conf.id in t.best_finisher:
            return t

def get_tasks(taskdic, tasks):
    """
    Map back a ray object to the conf/instance pair.
    :param taskdic: Nested dic of {conf: {instance: ray object}}
    :param tasks: List with ray objects that are running
    :return: List of [conf, instances] pairs that are currently running
    """
    running_tasks = []
    for conf, instance in taskdic.items():
        for instance_name, object in instance.items():
            if object in tasks:
                running_tasks.append([conf, instance_name])
    return running_tasks

def update_tasks(tasks, next_task, tournament, global_cache, ta_wrapper, scenario):
    """

    :param tasks: List of ray objects
    :param next_task: List of [conf, instance] pairs
    :param tournament: Tournament the next task is part of
    :param global_cache: Ray cache
    :param ta_wrapper:
    :param scenario:
    :return: Updated list of ray objects
    """
    for t in next_task:
        if t[1] is not None:
            # TODO need to change the wrapper to something more generic here
            task = tae_from_cmd_wrapper.remote(t[0], t[1], global_cache, ta_wrapper, scenario)
            tasks.append(task)
            # We also add the ray object id to the tournament to latter map the id back
            if t[0].id not in tournament.ray_object_store.keys():
                tournament.ray_object_store[t[0].id] = {t[1]: task}
            else:
                tournament.ray_object_store[t[0].id][t[1]] = task
    return tasks

def clear_logs():
    """
    Clear the logs
    """
    for folder in ['./selector/logs' ,'./selector/logs/ta_logs']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)


