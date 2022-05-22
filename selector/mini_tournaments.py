import logging
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import ray
import time


from scenario import Scenario
from pool import Configuration
from pointselector import RandomSelector
from ta_result_store import TargetAlgorithmObserver

from selector.point_gen import PointGen
from selector.random_point_generator import random_point

from tournament_dispatcher import MiniTournamentDispatcher
from tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, clear_logs

from tournament_monitor import Monitor
from tournament_performance import overall_best_update

from wrapper.glucose_wrapper import GLucoseWrapper
from wrapper.tap_sleep_wrapper import TAP_Sleep_Wrapper
from wrapper.tap_work_wrapper import TAP_Work_Wrapper
from instance_sets import InstanceSet


from aggr_capping import AggrMonitor
from instance_monitor import InstanceMonitor


def termination_check(termination_criterion, main_loop_start, total_runtime, total_tournament_number,
                      tournament_counter):
    """
    Check what termination criterion for the main tournament loop has been parsed and return true,
    if the criterion is not met yet.
    :param termination_criterion: Str. termination criterion for the tournament main loop
    :param main_loop_start: Int. Time of the start of the tournament main loop
    :param total_runtime: Int. Total runtime for the main loop, when the termination criterion is "total_runtime"
    :param total_tournament_number: Int. Total number of tournaments for the main loop,
                                    when the termination criterion is "total_tournament_number"
    :param tournament_counter: Int. Number of tournaments, that finished already
    :return: Bool. True, when the termination criterion is not met, False otherwise
    """
    if termination_criterion == "total_runtime":
        return time.time() - main_loop_start < total_runtime

    elif termination_criterion == "total_tournament_number":
        return tournament_counter < total_tournament_number

    else:
        return time.time() - main_loop_start < total_runtime


def offline_mini_tournament_configuration(scenario, ta_wrapper, logger):
    point_selector = RandomSelector()
    tournament_dispatcher = MiniTournamentDispatcher()
    global_cache = TargetAlgorithmObserver.remote()
    #monitor = Monitor.remote(1, global_cache, scenario.winners_per_tournament)
    monitor = InstanceMonitor.remote(1, global_cache)
    random_generator = PointGen(scenario, random_point)

    instance_selector = InstanceSet(scenario.instance_set, scenario.initial_instance_set_size, scenario.set_size)
    tasks = []
    tournaments = []
    tournament_counter = 0

    # creating the first tournaments and adding first conf/instance pairs to ray tasks
    for _ in range(scenario.number_tournaments):
        generated_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
        points_to_run = point_selector.select_points(generated_points, scenario.tournament_size, tournament_counter)

        instance_id, instances = instance_selector.get_subset(0)
        tournament, initial_assignments = tournament_dispatcher.init_tournament(global_cache, points_to_run,
                                                                                instances, instance_id)
        tournaments.append(tournament)
        global_cache.put_tournament_history.remote(tournament)
        tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    #starting the monitor
    global_cache.put_tournament_update.remote(tournaments)
    monitor.monitor.remote()

    logger.info(f"Initial Tournaments {tournaments}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")

    # TODO other convergence criteria DOTAC-36
    main_loop_start = time.time()

    while termination_check(scenario.termination_criterion, main_loop_start, scenario.total_runtime,
                            scenario.total_tournament_number, tournament_counter):

        logger.info("Starting main loop")
        if scenario.termination_criterion == "total_runtime":
            logger.info(f"The termination criterion is: {scenario.termination_criterion}")
            logger.info(f"The total runtime is: {scenario.total_runtime}")
        elif scenario.termination_criterion == "total_tournament_number":
            logger.info(f"The termination criterion is: {scenario.termination_criterion}")
            logger.info(f"The total number of tournaments is: {scenario.total_tournament_number}")
        else:
            logger.info(f"No valid termination criterion has been parsed. "
                        f"The termination criterion will be set to runtime.")
            logger.info(f"The total runtime is: {scenario.total_runtime}")

        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
        except ray.exceptions.WorkerCrashedError as e:
            logger.info('Crashed TA worker', time.ctime(), winner, e)
            continue

        result_conf, result_instance, cancel_flag = result[0], result[1], result[2]
        result_tournament = get_tournament_membership(tournaments, result_conf)

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        if cancel_flag:
            if result_tournament.ray_object_store[result_conf.id][result_instance] in tasks:
                tasks.remove(result_tournament.ray_object_store[result_conf.id][result_instance])
            logger.info(f"Canceled TA: {result_conf.id}, {result_instance}")
        else:
            result_time = ray.get(global_cache.get_results.remote())[result_conf.id][result_instance]
            logger.info(f"TA result: {result_conf.id}, {result_instance} {result_time}")

        # Update the tournament based on result
        result_tournament, tournament_stop = tournament_dispatcher.update_tournament(global_cache, tasks, result_conf,
                                                                                     result_tournament,
                                                                                     scenario.winners_per_tournament)

        global_cache.put_tournament_history.remote(result_tournament)
        logger.info(f"Result tournament update: Id: {result_tournament.id}"
                    f"Best finisher: {[c.id for c in result_tournament.best_finisher]}"
                    f", Worst finisher: {[c.id for c in result_tournament.worst_finisher]}"
                    f", Remaining configurations: {[c.id for c in result_tournament.configurations]} {tournament_stop}")

        if tournament_stop:
            print("Iteration:", time.time() - main_loop_start, tournament_counter)
            tournament_counter += 1

            # Generate and select
            generated_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            points_to_run = point_selector.select_points(generated_points, scenario.tournament_size - 1, tournament_counter)
            points_to_run = points_to_run + [result_tournament.best_finisher[0]]

            # Get the instances for the new tournament
            instance_id, instances = instance_selector.get_subset(result_tournament.instance_set_id + 1)

            # Create new tournament
            new_tournament, initial_assignments_new_tournament = tournament_dispatcher.init_tournament(global_cache,
                                                                                                       points_to_run,
                                                                                                       instances,
                                                                                                       instance_id)
            # Remove that old tournament
            tournaments.remove(result_tournament)

            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            tournaments.append(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache,  ta_wrapper, scenario)
            global_cache.put_tournament_history.remote(new_tournament)
            global_cache.put_tournament_update.remote(tournaments)

            logger.info(f"Final results tournament {result_tournament}")
            logger.info(f"New tournament {new_tournament}")
            logger.info(f"Initial Tasks of new tournament, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
        else:
            # If the tournament does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = tournament_dispatcher.next_tournament_run(global_cache, result_tournament, result_conf)
            tasks = update_tasks(tasks, next_task, result_tournament, global_cache, ta_wrapper, scenario)
            logger.info(f"Track new task {next_task}")
            logger.info(f"New Task {next_task}, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}, {result_tournament}")
            global_cache.put_tournament_update.remote(tournaments)

        overall_best_update(global_cache)

    print("DONE")
    logger.info("DONE")
    time.sleep(30)
    [ray.cancel(t) for t in not_ready]


if __name__ == "__main__":
    np.random.seed(42)
    clear_logs()

    logging.basicConfig( level=logging.INFO,
                        format='%(asctime)s %(message)s', handlers = [
        logging.FileHandler("./selector/logs/main.log"),
    ])

    logger = logging.getLogger(__name__)

    parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament": 2,
              "initial_instance_set_size": 3, "tournament_size": 4, "number_tournaments": 2, "total_tournament_number": 2,
              "total_runtime": 1200, "generator_multiple": 5, "set_size": 50,
              "termination_criterion": "total_runtime"}

    scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)
    # TODO this needs to come from the scenario?!
    #ta_wrapper = GLucoseWrapper()
    #ta_wrapper = TAP_Sleep_Wrapper()
    ta_wrapper = TAP_Work_Wrapper()

    # init
    #ray.init(address="auto")
    ray.init()

    logger.info("Ray info: {}".format(ray.cluster_resources()))
    logger.info("Ray nodes {}".format(ray.nodes()))
    logger.info("WD: {}".format(os.getcwd()))

    offline_mini_tournament_configuration(scenario, ta_wrapper, logger)

    ray.shutdown()






