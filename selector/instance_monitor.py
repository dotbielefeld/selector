"""This module contains the instance monitoring functionalities."""
import ray
import logging
import time


@ray.remote(num_cpus=1)
class InstanceMonitor:
    """
    Monitor whether the runtime of a configuration on an instance exceeds the
    best runtime of any configuration on that instance multiplied by a constant
    (delta_cap).  When a runtime exceeds this bound the configuration/instance
    pair is terminated. The terminated configuration/instance pairs are stored
    in the termination_history to avoid double killing

    Parameters
    ----------
    sleep_time : int
        Wake up and check whether runtime is exceeded
    cache : selector.ta_result_store.TargetAlgorithmObserver
        Stores all tournament related data.
    scenario : selector.scenario.Scenario
        AC scenario.
    delta_cap : int
        Constant the current best runtime for each instance is multiplied by.
    """
    def __init__(self, sleep_time, cache, scenario, delta_cap=1):
        self.sleep_time = sleep_time
        self.cache = cache
        self.tournaments = []
        self.termination_history = {}
        self.best_instance_results = {}
        self.delta_cap = delta_cap
        self.last_t_check = 0
        self.scenario = scenario

        logging.basicConfig(filename=f'{scenario.log_location}{scenario.log_folder}/inst_monitor.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def monitor(self):
        """        
        Monitors a tournament and terminates a configuration/instance pair if
        necessary.
        """
        logging.info("Starting monitor")
        worst_finisher = []

        while True:

            # Get results that are already available for ta runs
            start = time.time()
            results = ray.get(self.cache.get_results.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting results {dur}")

            # get starting times for each conf/instance
            start = time.time()
            start_time = ray.get(self.cache.get_start.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting start {dur}")

            # Get the current tournaments that are in the cache
            start = time.time()
            tournaments = ray.get(self.cache.get_tournament.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting tournaments {dur}")

            start = time.time()
            tournament_history = ray.get(self.cache.get_tournament_history.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting tournament history {dur}")

            current_instance_set = tournaments[0].instance_set_id
            tournaments_to_consider = len(tournaments)

            # finding the worst finisher for the tournaments with the most seen instances
            if len(tournament_history) != self.last_t_check and len(tournament_history) >= 2:
                last_tournaments = {}
                set_counter = {}
                for finished_tournament in tournament_history.values():
                    if finished_tournament.instance_set_id in [current_instance_set, current_instance_set+1, current_instance_set-1]:
                        last_tournaments[finished_tournament.id] = finished_tournament
                        set_counter[finished_tournament.id] = finished_tournament.instance_set_id

                tournament_ids_to_consider = sorted(set_counter)[:tournaments_to_consider]
                worst_finisher = [last_tournaments[x].best_finisher[-1] for x in tournament_ids_to_consider]
                self.last_t_check = len(tournament_history)

            # Creating a dictionary containing the best runtime for each instance
            for conf in worst_finisher:
                # need to figure out which tournamnet has the highest instance number and then wich conf the smallest runtime
                for instance in results[conf.id]:
                    if instance in self.best_instance_results and \
                            results[conf.id][instance] < self.best_instance_results[instance]:
                        self.best_instance_results[instance] = results[conf.id][instance]
                    elif instance not in self.best_instance_results:
                        self.best_instance_results[instance] = results[conf.id][instance]

            logging.info(f"best_instance_results: {self.best_instance_results}")

            for t in tournaments:
                if len(t.ray_object_store.keys()) >= 1:
                    for conf in t.configurations:
                        instances_conf_finished = []
                        if conf.id in list(results.keys()):
                            instances_conf_finished = list(results[conf.id].keys())
                        instances_conf_planned = list(t.ray_object_store[conf.id].keys())
                        instances_conf_still_runs = [i for i in instances_conf_planned if i not in instances_conf_finished]

                        # We kill a configuration/instance pair, when the runtime exceeds the current best runtime so far
                        # for that instance multiplied by delta_cap or when the configuration timed out
                        for instance in instances_conf_still_runs:
                            if conf.id in start_time and instance in start_time[conf.id] \
                                    and instance in self.best_instance_results:
                                instance_runtime = time.time() - start_time[conf.id][instance]
                                logging.info(
                                    f"Monitor kill check: conf.id: {conf.id}, "
                                    f"instance: {instance}, "
                                    f"instance_runtime: {instance_runtime}, "
                                    f"best_instance_runtime: {self.best_instance_results[instance]}")
                                if instance_runtime > \
                                        self.best_instance_results[instance] * self.delta_cap:
                                    if self.termination_check(conf.id, instance):
                                        logging.info(
                                            f"Monitor is killing: {conf} {instance} "
                                            f"with id: {t.ray_object_store[conf.id][instance]}")
                                        print(f"Monitor is killing: {time.ctime()} {t.ray_object_store[conf.id][instance]}")
                                        self.update_termination_history(conf.id, instance)
                                        [ray.cancel(t.ray_object_store[conf.id][instance])]
                                    else:
                                        continue

            time.sleep(self.sleep_time)

    def termination_check(self, conf_id, instance):
        """
        Check if we have killed a conf/instance pair already. Return True if
        we did not.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration to be checked.
        instance : str
            Name of instance to be checked.

        Returns
        -------
        bool
            False if configuration/instance pair killed, True else.
        """
        if conf_id not in self.termination_history:
            return True
        elif instance not in self.termination_history[conf_id]:
            return True
        else:
            return False

    def update_termination_history(self, conf_id, instance_id):
        """
        Stores termination events in history.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration that was killed.
        instance_id : str
            Name of instance the configuration was killed on.

        Returns
        -------
        bool
            False if configuration/instance pair killed, True else.
        """
        if conf_id not in self.termination_history:
            self.termination_history[conf_id] = []

        if instance_id not in self.termination_history[conf_id]:
            self.termination_history[conf_id].append(instance_id)
        else:
            logging.info(f"This should not happen: we kill something we already killed")
