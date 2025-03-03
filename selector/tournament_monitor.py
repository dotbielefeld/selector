"""In this module the tournament monitor is defined."""
import ray
import logging
import time
import numpy as np


from selector.tournament_performance import get_censored_runtime_for_instance_set,get_conf_time_out, get_runtime_for_instance_set_with_timeout


@ray.remote(num_cpus=1)
class Monitor:
    """
    Monitor whether the live total runtime of a running conf is exceeding the
    accumulated runtime of the worst finisher, given that we have already
    enough finisher. While up the monitor may kill multiple conf/instance
    pairs. To avoid killing a ta run twice, the monitor stores what it has
    already killed.

    Parameters
    ----------
    sleep_time : int
        Wake up and check whether runtime is exceeded
    cache : selector.ta_result_store.TargetAlgorithmObserver
        Stores all tournament related data.
    scenario : selector.scenario.Scenario
        AC scenario.
    """
    def __init__(self, sleep_time, cache, scenario):
        self.sleep_time = sleep_time
        self.cache = cache
        self.number_of_finisher = scenario.winners_per_tournament
        self.tournaments = []
        self.time_out = scenario.cutoff_time
        self.par = scenario.par

        logging.basicConfig(filename=f'{scenario.log_location}{scenario.log_folder}/monitor.log', level=logging.INFO,
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

            start = time.time()
            termination_history = ray.get(self.cache.get_termination_history.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting termination_history {dur}")

            for t in tournaments:
                # We can only start canceling runs if there are enough winners already
                if len(t.best_finisher) == self.number_of_finisher:
                    # Compare runtime to the worst best finisher
                    worst_best_finisher = t.best_finisher[-1]
                    runtime_worst_best_finisher = get_runtime_for_instance_set_with_timeout(results, worst_best_finisher.id,
                                                                                        t.instance_set, self.time_out, self.par)
                    # We need to compare each configuration that is still in the running to the worst finisher
                    for conf in t.configurations:
                        # Here we figured out which instances the conf is still running and which one it already finished
                        if conf.id in list(results.keys()):
                            instances_conf_finished = list(results[conf.id].keys())
                            conf_runtime_f = get_runtime_for_instance_set_with_timeout(results, conf.id, t.instance_set, self.time_out, self.par)
                        else:
                            instances_conf_finished = []
                            conf_runtime_f = 0
                        instances_conf_planned = list(t.ray_object_store[conf.id].keys())
                        instances_conf_still_runs = [c for c in instances_conf_planned if c not in instances_conf_finished]

                        # The runtime of a conf is the time it took to finish instances plus the time spend running but
                        # not finishing the running instances
                        # if i in list(start_time[conf.id].keys()): is a bit hack: it might be the case that the main
                        # process things a conf/instance pair is running but the cache has not recived a start time and
                        # thus the conf/instance is in a transition. That conf instance then has not runtime yet
                        # (or at least very very little) so we ignore it for the cancel computation
                        conf_runtime_p = sum([(time.time() - start_time[conf.id][i]) for i in instances_conf_still_runs if i in list(start_time[conf.id].keys())])
                        conf_runtime = conf_runtime_f + conf_runtime_p
                        #conf_time_out = get_conf_time_out(results, conf.id, t.instance_set)

                        logging.info(f"Monitor kill check,{conf.id} {conf_runtime}, {runtime_worst_best_finisher}"
                                     f"{worst_best_finisher.id,}, {[m. id for m in t.configurations]}")

                        if conf_runtime > runtime_worst_best_finisher:# or conf_time_out:
                            # We can only kill still running tasks
                            for i in instances_conf_still_runs:
                                # We check if we have killed the conf/instance pair before.
                                if self.termination_check(conf.id, i, termination_history):
                                    logging.info(f"Monitor is killing: {conf} {i} with id: {t.ray_object_store[conf.id][i]}")
                                    print(f"Monitor is killing:{time.ctime()} {t.ray_object_store[conf.id][i]}")
                                    # In case we kill we store that we have killed
                                    self.cache.put_termination_history.remote(conf.id, i)
                                    [ray.cancel(t.ray_object_store[conf.id][i])]
                                else:
                                    continue
            time.sleep(self.sleep_time)

    def termination_check(self, conf_id, instance, termination_history):
        """
        Check if we have killed a conf/instance pair already. Return True if
        we did not.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration to be checked.
        instance : str
            Name of instance to be checked.
        termination_history : dict
            Record of configuration/instance terminations.

        Returns
        -------
        bool
            False if configuration/instance pair killed, True else.
        """
        if conf_id not in termination_history:
            return True
        elif instance not in termination_history[conf_id]:
            return True
        else:
            return False
