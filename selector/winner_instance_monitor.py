"""In this module the instance monitor is defined."""
import ray
import logging
import time
import numpy as np


@ray.remote(num_cpus=1)
class WinnerInstanceMonitor:
    """
    Monitor whether the runtime of a configuration on an instance exceeds the
    worst runtime of a tournament winning configuration on that instance
    multiplied by a constant (delta_cap). When a runtime exceeds this bound
    the configuration/instance pair is terminated.
    The terminated configuration/instance pairs are stored in the
    termination_history to avoid double killing

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
        self.instance_results = {}
        self.delta_cap = delta_cap

        logging.basicConfig(filename=f'{scenario.log_location}{scenario.log_folder}winner_inst_monitor.log',
                            level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def monitor(self):
        """
        Monitors a tournament and terminates a configuration/instance pair if
        necessary.
        """
        logging.info("Starting monitor")

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

            # List with the winners of the tournaments currently running
            winners = [t.best_finisher[0] for t in tournaments if t.best_finisher]
            logging.info(f"Current tournament winners: {winners}")

            # We create a dict with the results for the instances for the tournament winners
            # The worst time by a winner for an instance, that did not time out, gets stored in the dict as a benchmark
            # for the instance based capping.
            self.instance_results = {}
            for conf in winners:
                for instance in results[conf.id]:
                    if instance in self.instance_results and results[conf.id][instance] > \
                            self.instance_results[instance] and not np.isnan(results[conf.id][instance]):
                        self.instance_results[instance] = results[conf.id][instance]
                    elif instance not in self.instance_results and not np.isnan(results[conf.id][instance]):
                        self.instance_results[instance] = results[conf.id][instance]

            logging.info(f"Results of the current tournament winners: {self.instance_results}")

            for t in tournaments:
                for conf in t.configurations:
                    instances_conf_finished = []
                    if conf.id in list(results.keys()):
                        instances_conf_finished = list(results[conf.id].keys())

                    instances_conf_planned = list(t.ray_object_store[conf.id].keys())
                    instances_conf_still_runs = [i for i in instances_conf_planned if i not in instances_conf_finished]

                    # We kill a configuration/instance pair, when the runtime exceeds the benchmark runtime
                    # for that instance multiplied by delta_cap
                    for instance in instances_conf_still_runs:
                        if conf.id in start_time and instance in start_time[conf.id] \
                                and instance in self.instance_results:
                            instance_runtime = time.time() - start_time[conf.id][instance]
                            logging.info(
                                f"Monitor kill check: conf.id: {conf.id}, "
                                f"instance: {instance}, "
                                f"instance_runtime: {instance_runtime}, "
                                f"best_instance_runtime: {self.instance_results[instance]}")
                            if instance_runtime > self.instance_results[instance] * self.delta_cap \
                                    and self.termination_check(conf.id, instance):
                                logging.info(
                                    f"Monitor is killing: {conf} {instance} "
                                    f"with id: {t.ray_object_store[conf.id][instance]}")
                                print(f"Monitor is killing: {time.ctime()} {t.ray_object_store[conf.id][instance]}")
                                self.update_termination_history(conf.id, instance)
                                logging.info(f"termination_history: {self.termination_history}")
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
            logging.info(
                "This should not happen: we kill something we already killed")
