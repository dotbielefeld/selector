import numpy as np
import ray
import uuid
import time

from selector.pool import Tournament
from selector.tournament_performance import get_total_runtime_for_instance_set, get_instances_no_results, get_conf_time_out

class MiniTournamentDispatcher:

    def init_tournament(self, cache, configurations, instance_partition, instance_partition_id):
        """
        Create a new tournament out of the given configurations and list of instances.
        :param cache: Results cache.
        :param configurations: List. Configurations for the tournament
        :param instance_partition: List. List of instances
        :param instance_partition_id: Id of the instance set.
        :return: Tournament, first conf/instance assignment to run
        """

        results = ray.get(cache.get_results.remote())

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

            configurations_not_run_on_all = configurations
            configurations_not_run_on_all.remove(most_run_conf)

            extra_instances = instance_partition
            extra_instances.remove(first_instance)

            extra_assignment = [np.random.choice(configurations_not_run_on_all), np.random.choice(extra_instances)]
            initial_instance_conf_assignments = [[conf, first_instance] for conf in configurations_not_run_on_all] \
                                                + [extra_assignment]

            best_finisher = [most_run_conf]

        configuration_ids = [c.id for c in configurations]
        return Tournament(uuid.uuid4(), best_finisher, [], configurations, configuration_ids, {}, instance_partition,
                          instance_partition_id), \
               initial_instance_conf_assignments


    def update_tournament(self, cache, finished_conf, tournament, number_winner):
        """
        Given a finishing conf we update the tournament if necessary. I.e the finishing conf has seen all instances of
        the tournament. In that case, it is moved either to the best or worst finishes. best finishers are ordered.
        Worst finishers are not
        :param cache: Ray cache object.
        :param finished_conf: Configuration that finished or was canceled
        :param tournament: Tournament the finish conf was a member of
        :param number_winner: Int that determines the number of winners per tournament
        :return: updated tournament, stopping signal
        """
        results = ray.get(cache.get_results.remote())
        conf_time_out = get_conf_time_out(results, finished_conf.id, tournament.instance_set)
        evaluated_instances = results[finished_conf.id].keys()

        # A conf can only become a best finisher if it has seen all instances of the tournament
        if set(evaluated_instances) == set(tournament.instance_set):
            # We can than remove the conf from further consideration
            tournament.configurations.remove(finished_conf)
            finished_conf_runtime = get_total_runtime_for_instance_set(results, finished_conf.id,
                                                                       tournament.instance_set)

            # If there are already some best finisher we need to compare the conf to them
            if len(tournament.best_finisher) > 0:
                # We assume that the finishers in the set are ordered according to their runtime
                for bfi in range(len(tournament.best_finisher)):
                    bfr = get_total_runtime_for_instance_set(results, tournament.best_finisher[bfi].id,
                                                             tournament.instance_set)

                    # If the conf is better than one best finisher we insert it
                    if finished_conf_runtime <= bfr and not conf_time_out:
                        tournament.best_finisher.insert(bfi, finished_conf)
                        if finished_conf in tournament.worst_finisher:
                            tournament.worst_finisher.remove(finished_conf)
                        # If we have too many best finishers we cut off the excess
                        if len(tournament.best_finisher) > number_winner:
                            transition =  number_winner - len(tournament.best_finisher)
                            tournament.worst_finisher = tournament.worst_finisher + tournament.best_finisher[transition:]
                            tournament.best_finisher = tournament.best_finisher[: transition]
                            break
                    # We also add a conf to best finishers if we have not enough
                    elif len(tournament.best_finisher) < number_winner:
                        tournament.best_finisher.append(finished_conf)
                        break
                    # If the conf is not better it is a worst finisher
                    elif finished_conf not in tournament.worst_finisher:
                        tournament.worst_finisher.append(finished_conf)
            else:
                tournament.best_finisher.append(finished_conf)


        # If there are no configurations left we end the tournament
        if len(tournament.configurations) == 0:
            stop = True
        else:
            stop = False

        return tournament, stop

    def next_tournament_run(self, cache, tournament, finished_conf):
        """
        Decided which conf/instance pair to run next. Rule: If the configuration that has just finished was not killed
        nor saw all instances, it is assigned a new instance at random. Else, the configuration with the lowest runtime
        so far is selected.
        :param cache: Ray cache
        :param tournament: The tournament we opt to create a new task for
        :param finished_conf: Configuration that just finished before
        :return: configuration, instance pair to run next
        """
        results = ray.get(cache.get_results.remote())
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




