import logging
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import ray


from scenario import Scenario
from pool import Configuration
from pointselector import RandomSelector
from ta_result_store import TargetAlgorithmObserver
from generator import generate

from tournament_dispatcher import MiniTournamentDispatcher
from tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, clear_logs
from tournament_monitor import monitor
from tournament_performance import overall_best_update

from wrapper.glucose_wrapper import GLucoseWrapper
from wrapper.tap_sleep_wrapper import TAP_Sleep_Wrapper
from wrapper.tap_work_wrapper import TAP_Work_Wrapper
from instance_sets import InstanceSet




def offline_mini_tournament_configuration(scenario, ta_wrapper, logger):
    point_selector = RandomSelector()
    tournament_dispatcher = MiniTournamentDispatcher()
    global_cache = TargetAlgorithmObserver.remote()

    instance_selector = InstanceSet(scenario.instance_set, scenario.initial_instance_set_size)

    tasks = []
    tournaments = []
    tournament_counter = 0

    # creating the first tournaments and adding first conf/instance pairs to ray tasks
    for _ in range(scenario.number_tournaments):
        generated_points = [generate(scenario) for _ in range(scenario.tournament_size*5)]
        points_to_run = point_selector.select_points(generated_points, scenario.tournament_size, tournament_counter)

        instance_id, instances = instance_selector.get_subset(0)
        tournament, initial_assignments = tournament_dispatcher.init_tournament(global_cache, points_to_run,
                                                                                instances, instance_id)
        tournaments.append(tournament)
        global_cache.put_tournament_history.remote(tournament)
        tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    #adding the monitor to ray tasks
    monitor_task = monitor.remote(1, tournaments, global_cache, scenario.winners_per_tournament)
    tasks.insert(0, monitor_task)

    logger.info(f"Initial Tournaments {tournaments}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")

    # TODO other convergence criteria DOTAC-36
    while tournament_counter < scenario.total_tournament_number:
        logger.info("Starting main loop")
        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        result = ray.get(winner)[0]
        result_conf, result_instance = result[0], result[1]
        result_tournament = get_tournament_membership(tournaments, result_conf)

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        if len(result) == 3:
            if result_tournament.ray_object_store[result_conf.id][result_instance] in tasks:
                tasks.remove(result_tournament.ray_object_store[result_conf.id][result_instance])
            logger.info(f"Canceled TA: {result_conf.id}, {result_instance}")
        else:
            ray.cancel(monitor_task, recursive=False)
            tasks.remove(monitor_task)
            result_time = ray.get(global_cache.get_results.remote())[result_conf.id][result_instance]

            logger.info(f"TA result: {result_conf.id}, {result_instance} {result_time}")

        # Update the tournament based on result
        result_tournament, tournament_stop = tournament_dispatcher.update_tournament(global_cache, result_conf,
                                                                                     result_tournament,
                                                                                     scenario.winners_per_tournament)

        global_cache.put_tournament_history.remote(result_tournament)
        logger.info(f"Result tournament update: Id: {result_tournament.id}"
                    f"Best finisher: {[c.id for c in result_tournament.best_finisher]}"
                    f", Worst finisher: {[c.id for c in result_tournament.worst_finisher]}"
                    f", Remaining configurations: {[c.id for c in result_tournament.configurations]} {tournament_stop}")

        if tournament_stop:
            tournament_counter += 1

            # Generate and select
            generated_points = [generate(scenario) for _ in range(scenario.tournament_size*5)]
            points_to_run = point_selector.select_points(generated_points, scenario.tournament_size-1, tournament_counter)
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
            global_cache.put_tournament_history.remote(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache, ta_wrapper, scenario)

            logger.info(f"Final results tournament {result_tournament}")
            logger.info(f"New tournament {new_tournament}")
            logger.info(f"Initial Tasks of new tournament, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
        else:
            # If the tournament does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = tournament_dispatcher.next_tournament_run(global_cache, result_tournament, result_conf)
            tasks = update_tasks(tasks, next_task, result_tournament, global_cache, ta_wrapper, scenario)
            logger.info(f"New Task {next_task}, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}, {result_tournament}" )

        # After each ray task we cancel and restart the monitor regardless of whether it terminated or not
        monitor_task = monitor.remote(1, tournaments, global_cache, scenario.winners_per_tournament)
        tasks.insert(0, monitor_task)
        overall_best_update(global_cache)


    print("DONE")
    [ray.cancel(t) for t in not_ready]


if __name__ == "__main__":
    np.random.seed(42)
    clear_logs()

    logging.basicConfig( level=logging.INFO,
                        format='%(asctime)s %(message)s', handlers = [
        logging.FileHandler("./selector/logs/main.log"),
    ])

    logger = logging.getLogger(__name__)


    parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament" : 1,
            "initial_instance_set_size": 2, "tournament_size": 2, "number_tournaments": 1, "total_tournament_number":2 }

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







