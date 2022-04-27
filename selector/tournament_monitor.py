import ray
import logging
import time
import numpy as np


from tournament_performance import get_censored_runtime_for_instance_set,get_conf_time_out

def termination_check(conf_id, instance, termination_history):
    if conf_id not in termination_history:
        return True
    elif instance not in termination_history[conf_id]:
        return True
    else:
        return False

@ray.remote(num_cpus=1)
def monitor(sleep_time, tournaments, cache, number_of_finisher):
    """
    Monitor whether the live total runtime of a running conf is exceeding the accumulated runtime of the worst finisher,
     given that we have already enough finisher. While up the monitor may kill multiple conf/instance pairs. Since the
     monitor itself may be canceled in case the main loop receives some ta results we need to store what we have already
     killed to avoid kill a conf/instance twice.
    :param sleep_time: Int. Wake up and check whether runtime is exceeded
    :param tournaments: List with the current tournaments
    :param cache: Ray cache
    :param number_of_finisher: Int.
    :return: conf/instance that are killed.
    """
    logging.basicConfig(filename=f'./selector/logs/monitor.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')
    try:
        logging.info("Starting monitor")
        while True:
            # Todo mesure time here
            start = time.time()
            results = ray.get(cache.get_results.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting results {dur}")

            # get starting times for each conf/instance
            start = time.time()
            start_time = ray.get(cache.get_start.remote())
            dur = time.time() - start
            logging.info(f"Monitor getting start {dur}")

            termination_history = ray.get(cache.get_ta_termination.remote())

            for t in tournaments:
                # We can only start canceling runs if there are enough winners already
                if len(t.best_finisher) == number_of_finisher:
                    # Compare runtime to the worst best finisher
                    worst_best_finisher = t.best_finisher[-1]
                    runtime_worst_best_finisher = get_censored_runtime_for_instance_set(results, worst_best_finisher.id,
                                                                                        t.instance_set)
                    # We need to compare each configuration that is still in the running to the worst finisher
                    for conf in t.configurations:
                        # Here we figured out which instances the conf is still running and which one it already finished
                        if conf.id in list(results.keys()):
                            instances_conf_finished = list(results[conf.id].keys())
                            conf_runtime_f = get_censored_runtime_for_instance_set(results, conf.id, t.instance_set)
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
                        conf_time_out = get_conf_time_out(results, conf.id, t.instance_set)

                        logging.info(f"Monitor kill check,{conf.id} {conf_runtime}, {runtime_worst_best_finisher}"
                                     f"{worst_best_finisher.id,},{conf_time_out}, {[m. id for m in t.configurations]}")

                        # In one loop we only kill one instance the conf is still running on. In case the conf runs on multiple
                        # instances we sleep and then kill
                        # We also kill in case there has been a time out recorded for the conf
                        if conf_runtime > runtime_worst_best_finisher or conf_time_out:
                            for i in instances_conf_still_runs:
                                tc = termination_check(conf.id, i, termination_history)
                                if tc:
                                    logging.info(f"Monitor is killing: {conf} {i} with id: {t.ray_object_store[conf.id][i]}")
                                    print(f"Monitor is killing:{time.ctime()} {t.ray_object_store[conf.id][i]}")
                                    cache.put_ta_termination.remote(conf.id, i)
                                    [ray.cancel(t.ray_object_store[conf.id][i])]
                                else:
                                    continue
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Monitor is killed")



