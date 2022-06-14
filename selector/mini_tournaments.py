import logging
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import ray
import time


from scenario import Scenario
from pool import Configuration
from pointselector import RandomSelector, HyperparameterizedSelector
from ta_result_store import TargetAlgorithmObserver

from selector.point_gen import PointGen
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator

from tournament_dispatcher import MiniTournamentDispatcher
from tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, termination_check
from log_setup import clear_logs, log_termination_setting

from tournament_monitor import Monitor
from tournament_performance import overall_best_update

from wrapper.tap_sleep_wrapper import TAP_Sleep_Wrapper
from wrapper.tap_work_wrapper import TAP_Work_Wrapper
from instance_sets import InstanceSet


from instance_monitor import InstanceMonitor




def offline_mini_tournament_configuration(scenario, ta_wrapper, logger):
    log_termination_setting(logger, scenario)

    point_selector = RandomSelector()
    hp_seletor = HyperparameterizedSelector()
    tournament_dispatcher = MiniTournamentDispatcher()
    global_cache = TargetAlgorithmObserver.remote()
    monitor = Monitor.remote(1, global_cache, scenario.winners_per_tournament)
    #monitor = InstanceMonitor.remote(1, global_cache)
    random_generator = PointGen(scenario, random_point)
    default_point_generator = PointGen(scenario, default_point)
    vg_point_generator = PointGen(scenario, variable_graph_point)
    lhc_point_generator = PointGen(scenario, lhc_points)

    instance_selector = InstanceSet(scenario.instance_set, scenario.initial_instance_set_size, scenario.set_size)
    tasks = []
    tournaments = []
    tournament_counter = 0
    results = ray.get(global_cache.get_results.remote())

    # creating the first tournaments and adding first conf/instance pairs to ray tasks
    for _ in range(scenario.number_tournaments):
        generated_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]

        points_to_run = point_selector.select_points(generated_points, scenario.tournament_size, tournament_counter)


        instance_id, instances = instance_selector.get_subset(0)
        tournament, initial_assignments = tournament_dispatcher.init_tournament(results, points_to_run,
                                                                                instances, instance_id)
        tournaments.append(tournament)
        global_cache.put_tournament_history.remote(tournament)
        global_cache.put_tournament_update.remote(tournament)
        tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    #starting the monitor
    #global_cache.put_tournament_update.remote(tournaments)
    monitor.monitor.remote()

    logger.info(f"Initial Tournaments {tournaments}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")

    main_loop_start = time.time()
    epoch = 0
    max_epochs = 256

    while termination_check(scenario.termination_criterion, main_loop_start, scenario.total_runtime,
                            scenario.total_tournament_number, tournament_counter):

        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
            result_conf, result_instance, cancel_flag = result[0], result[1], result[2]
        # Some time a ray worker may crash. We handel that here. I.e if the TA did not run to the end, we reschedule
        except ray.exceptions.WorkerCrashedError as e:
            logger.info(f'Crashed TA worker, {time.ctime()}, {winner}, {e}')
            print("Crashed TA worker")
            # Figure out which tournament conf. belongs to
            for t in tournaments:
                conf_instance = get_tasks(t.ray_object_store, winner)
                if  len(conf_instance) != 0:
                    tournament_of_c_i = t
                    break

            conf = [conf for conf in tournament_of_c_i.configurations if conf.id == conf_instance[0][0]][0]
            instance = conf_instance[0][1]
            # We check if we have killed the conf and only messed up the termination of the process
            termination_history = ray.get(global_cache.get_termination_history.remote())
            # TODO we probably need to check, that we really killed the process..
            if conf.id in termination_history.keys() and instance in termination_history[conf.id]:
                result_conf = conf
                result_instance = instance
                cancel_flag = True
                global_cache.put_result.remote(result_conf.id, result_instance, np.nan)
                logger.info(f"Canceled task with no return: {result_conf}, {result_instance}")
            else: #got no results: need to rescheulde
                next_task = [[conf, instance]]
                tasks = update_tasks(tasks, next_task, tournament_of_c_i, global_cache, ta_wrapper, scenario)
                logger.info(f"We have no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
                continue

        except ray.exceptions.TaskCancelledError as e:
            logger.info(f'This should only happen if the tournament are bigger then the number of cpu, {e}')

        results = ray.get(global_cache.get_results.remote())
        result_tournament = get_tournament_membership(tournaments, result_conf)

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        if cancel_flag:
            if result_tournament.ray_object_store[result_conf.id][result_instance] in tasks:
                tasks.remove(result_tournament.ray_object_store[result_conf.id][result_instance])
            logger.info(f"Canceled TA: {result_conf.id}, {result_instance}")
        else:
            result_time = results[result_conf.id][result_instance]
            logger.info(f"TA result: {result_conf.id}, {result_instance} {result_time}")

        # Update the tournament based on result
        result_tournament, tournament_stop = tournament_dispatcher.update_tournament(results, tasks, result_conf,
                                                                                     result_tournament,
                                                                                     scenario.winners_per_tournament,
                                                                                     scenario.cutoff_time, scenario.par)

        global_cache.put_tournament_history.remote(result_tournament)
        logger.info(f"Result tournament update: Id: {result_tournament.id}"
                    f"Best finisher: {[c.id for c in result_tournament.best_finisher]}"
                    f", Worst finisher: {[c.id for c in result_tournament.worst_finisher]}"
                    f", Remaining configurations: {[c.id for c in result_tournament.configurations]} {tournament_stop}")

        if tournament_stop:
            print("Iteration:", time.time() - main_loop_start, tournament_counter)
            tournament_counter += 1

            # Generate and select

            #generated_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            #points_to_run = point_selector.select_points(generated_points, scenario.tournament_size - 1, tournament_counter)

            random_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            # points_to_run = point_selector.select_points(generated_points, scenario.tournament_size-1, tournament_counter)
            default_ps = [default_point_generator.point_generator()]
            hist = ray.get(global_cache.get_tournament_history.remote())
            vg_points = [vg_point_generator.point_generator(
                         mode=Mode.random, alldata=hist, lookback=1)
                         for _ in range(
                         scenario.tournament_size *
                         scenario.generator_multiple)]
            lhc_ps = lhc_point_generator.point_generator(
                n_samples=(scenario.tournament_size *
                           scenario.generator_multiple),
                lhs_type=LHSType.centered,
                criterion=Criterion.maximin)

            generated_points = random_points + default_ps + \
                vg_points + lhc_ps

            weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            weights = [weights for x in range(len(generated_points))]
            weights = np.array(weights)

            fg = FeatureGenerator()
            features = fg.static_feature_gen(generated_points, epoch,
                                             max_epochs)

            points_to_run = \
                hp_seletor.select_points(scenario, generated_points,
                                         scenario.tournament_size - 1,
                                         epoch, max_epochs, features, weights,
                                         max_evals=100)

            points_to_run = points_to_run + [result_tournament.best_finisher[0]]

            # Get the instances for the new tournament
            instance_id, instances = instance_selector.get_subset(result_tournament.instance_set_id + 1)

            # Create new tournament
            new_tournament, initial_assignments_new_tournament = tournament_dispatcher.init_tournament(results,
                                                                                                       points_to_run,
                                                                                                       instances,
                                                                                                       instance_id)
            # Remove that old tournament
            tournaments.remove(result_tournament)

            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            tournaments.append(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache,  ta_wrapper, scenario)
            global_cache.put_tournament_history.remote(new_tournament)

            global_cache.put_tournament_update.remote(new_tournament)
            global_cache.remove_tournament.remote(result_tournament)
           # global_cache.put_tournament_update.remote(tournaments)

            logger.info(f"Final results tournament {result_tournament}")
            logger.info(f"New tournament {new_tournament}")
            epoch += 1
        else:
            # If the tournament does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = tournament_dispatcher.next_tournament_run(results, result_tournament, result_conf)
            tasks = update_tasks(tasks, next_task, result_tournament, global_cache, ta_wrapper, scenario)
            logger.info(f"New Task {next_task}, {result_tournament}")
            #global_cache.put_tournament_update.remote(tournaments)
            global_cache.put_tournament_update.remote(result_tournament)

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

    parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament": 1,
              "initial_instance_set_size": 2, "tournament_size": 2, "number_tournaments": 1, "total_tournament_number": 3,
              "total_runtime": 1200, "generator_multiple": 5, "set_size": 50,
              "termination_criterion": "total_tournament_number", "par": 1, "ta_pid_name": "glucose-simp"}

    scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)
    # TODO this needs to come from the scenario?!
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






