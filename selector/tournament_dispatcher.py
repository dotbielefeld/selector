"""This module contains functions for managing tournaments."""
import numpy as np
import ray
import uuid
import time
import copy

from selector.pool import Tournament
from selector.tournament_performance import (
    get_censored_runtime_for_instance_set,
    get_instances_no_results,
    get_conf_time_out,
    get_runtime_for_instance_set_with_timeout
)


class MiniTournamentDispatcher:
    """
    Dispatches tournaments.
    """

    def init_tournament(self, results, configurations, instance_partition, instance_partition_id):
        """
        Create a new tournament from the given configurations and list of instances.

        Parameters
        ----------
        results : dict
            Dic of results: {conf_id: {instance: runtime}}.
        configurations : list
            Configurations for the tournament.
        instance_partition : list
            List of instances.
        instance_partition_id : int
            ID of the instance set.

        Returns
        -------
        tuple
            - selector.pool.Tournament,
              Tournament object
            - list,
              The first configuration/instance assignment to run.
        """
        # Get the configuration that has seen the most instances before
        conf_instances_ran = []
        most_run_conf = None
        for conf in configurations:
            if conf.id in list(results.keys()):
                conf_instances_ran = list(results[conf.id].keys())
                most_run_conf = conf

        # Get instances the conf with the most runs has not been run on before
        possible_first_instances = [i for i in instance_partition if i not in conf_instances_ran]

        # If there are instances the conf with the most runs has not seen we select on of them to be the first instance
        # all confs should be run on
        if possible_first_instances:
            first_instance = np.random.choice(possible_first_instances)
            initial_instance_conf_assignments = [[conf, first_instance] for conf in configurations]
            best_finisher = []
        # An empty list of possible instances means that the conf with the most runs has seen all instances in the
        # instance set. In that case we can choose any instance for the confs that have not seen all instances.
        # We also have a free core then to which we assign a extra conf/instance pair where both are chosen at random
        else:
            first_instance = np.random.choice(instance_partition)

            #configurations_not_run_on_all = copy.deepcopy(configurations)
            configurations_not_run_on_all = configurations
            configurations_not_run_on_all.remove(most_run_conf)

            extra_instances = copy.deepcopy(instance_partition)
            extra_instances.remove(first_instance)

            extra_assignment = [np.random.choice(configurations_not_run_on_all), np.random.choice(extra_instances)]
            initial_instance_conf_assignments = [[conf, first_instance] for conf in configurations_not_run_on_all] \
                                                + [extra_assignment]

            best_finisher = [most_run_conf]

        configuration_ids = [c.id for c in configurations] + [b.id for b in best_finisher if len(best_finisher) >= 1]

        return Tournament(uuid.uuid4(), best_finisher, [], configurations, configuration_ids, {}, instance_partition,
                          instance_partition_id), \
               initial_instance_conf_assignments


    def update_tournament(self, results, tasks, finished_conf, tournament, number_winner, time_out, par_penalty):
        """
        Update the tournament based on a finished configuration.

        If the finished configuration has seen all instances of the tournament, it is moved 
        either to the best or worst finishers. Best finishers are ordered, while worst 
        finishers are not.

        Parameters
        ----------
        results : dict
            Dic of results: {conf_id: {instance: runtime}}.
        finished_conf : selector.pool.Configuration
            Configuration that finished or was canceled.
        tournament : selector.pool.Tournament
            Tournament the finished configuration was a member of.
        number_winner : int
            Determines the number of winners per tournament.

        Returns
        -------
        tuple
            - selector.pool.Tournament,
              Updated tournament 
            - bool,
              stopping signal.
        """
        evaluated_instances = results[finished_conf.id].keys()
        bfi_add = False

        # We figure out if there are still tasks the finished configuration is still running on for which we have a results
        # but have not returned through a ray.wait()
        still_running_task_for_conf = [sr for sr in tasks if sr in list(tournament.ray_object_store[finished_conf.id].values())]

        # A conf can only become a best finisher if it has seen all instances of the tournament and is not running any
        # other conf/instance pairs. i.e the result we process here is the last one
        if set(evaluated_instances) == set(tournament.instance_set) and len(still_running_task_for_conf) == 0:
            # We can than remove the conf from further consideration
            if finished_conf in tournament.configurations:
                tournament.configurations.remove(finished_conf)

            finished_conf_runtime_mean = get_runtime_for_instance_set_with_timeout(results, finished_conf.id,
                                                                                   tournament.instance_set, time_out, par_penalty) / len(tournament.instance_set)

            # If there are already some best finisher we need to compare the conf to them
            if len(tournament.best_finisher) > 0:
                # We assume that the finishers in the set are ordered according to their runtime
                for bfi in range(len(tournament.best_finisher)):
                    bfrm = get_runtime_for_instance_set_with_timeout(results, tournament.best_finisher[bfi].id,
                                                                     tournament.instance_set, time_out, par_penalty) / len(tournament.instance_set)

                    # If the conf is better than one best finisher we insert it
                    if finished_conf_runtime_mean <= bfrm:
                        tournament.best_finisher.insert(bfi, finished_conf)
                        bfi_add = True
                        if finished_conf in tournament.worst_finisher:
                            tournament.worst_finisher.remove(finished_conf)
                        # If we have too many best finishers we cut off the excess
                        if len(tournament.best_finisher) > number_winner:
                            transition =  number_winner - len(tournament.best_finisher)
                            tournament.worst_finisher = tournament.worst_finisher + tournament.best_finisher[transition:]
                            tournament.best_finisher = tournament.best_finisher[: transition]
                            break
                # We also add a conf to best finishers if we have not enough
                if len(tournament.best_finisher) < number_winner and not bfi_add:
                    tournament.best_finisher.append(finished_conf)

                # If the conf is not better it is a worst finisher
                elif finished_conf not in tournament.worst_finisher and not bfi_add:
                    tournament.worst_finisher.append(finished_conf)
            else:
                tournament.best_finisher.append(finished_conf)

        # If there are no configurations left we end the tournament
        if len(tournament.configurations) == 0:
            stop = True
        else:
            stop = False

        return tournament, stop

    def next_tournament_run(self, results, tournament, finished_conf):
        """
        Decide which configuration/instance pair to run next.

        Rule: If the configuration that has just finished was neither killed nor has seen 
        all instances, it is assigned a new instance at random. Otherwise, the configuration 
        with the lowest runtime so far is selected.

        Parameters
        ----------
        results : dict
            Dic of results: {conf_id: {instance: runtime}}.
        tournament : selector.pool.Tournament
            The tournament for which a new task is to be created.
        finished_conf : selector.pool.Configuration
            Configuration that just finished.

        Returns
        -------
        list
            Configuration and instance pair to run next.
        """

        next_possible_conf = {}

        # For each conf still in the running we need to figure out on which instances it already ran or is still
        # running on to get for each conf the instances it still can run on
        for conf in tournament.configurations:
            already_run = get_instances_no_results(results, conf.id, tournament.instance_set)

            not_running_currently = get_instances_no_results(tournament.ray_object_store, conf.id,
                                                             tournament.instance_set)
            not_running_currently = [c for c in not_running_currently if c in already_run]

            if len(not_running_currently) > 0:
                next_possible_conf[conf.id] = not_running_currently
        # If there are no configuration that need to see new instances we create a dummy task to give the still running
        # conf/instance pairs time to finish.
        if len(next_possible_conf) == 0:
            configuration = None
            next_instance = None
        else:
        # If the previous run conf has not seen all instances and did not time out it is selected to run again
            if finished_conf.id in list(next_possible_conf.keys()):
                next_conf_id = finished_conf.id
            else: #Select the configuration with the lowest mean runtime
                mean_rt_store = {}
                for conf in next_possible_conf.keys():
                    if conf in results.keys():
                        conf_rt = list(results[conf].values())
                        mean_rt_store[conf] = sum(conf_rt) / len(conf_rt)
                if mean_rt_store:
                    next_conf_id = min(mean_rt_store, key=mean_rt_store.get)
                # In case we have no results for any of the remaining configuration we sample
                else:
                    next_conf_id = np.random.choice(list(next_possible_conf.keys()))

            configuration = [c for c in tournament.configurations if c.id == next_conf_id][0]
            next_possible_instance = next_possible_conf[next_conf_id]
            next_instance = np.random.choice(next_possible_instance)

        return [[configuration, next_instance]]