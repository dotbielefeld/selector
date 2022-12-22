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
from ta_execution import dummy_task

from selector.point_gen import PointGen
from selector.pool import Status, Surrogates
from selector.random_point_generator import random_point
from selector.default_point_generator import default_point
from selector.variable_graph_point_generator import variable_graph_point, Mode
from selector.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator
from selector.surrogates.surrogates import SurrogateManager
# from selector.surrogates.surrogates import SurrogateManager

from tournament_dispatcher import MiniTournamentDispatcher
from tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, termination_check, get_get_tournament_membership_with_ray_id
from log_setup import clear_logs, log_termination_setting, check_log_folder, save_latest_logs

from tournament_monitor import Monitor
from tournament_performance import overall_best_update, get_instances_no_results

from wrapper.tap_sleep_wrapper import TAP_Sleep_Wrapper
from wrapper.tap_work_wrapper import TAP_Work_Wrapper
from instance_sets import InstanceSet


from instance_monitor import InstanceMonitor




def offline_mini_tournament_configuration(scenario, ta_wrapper, logger):
    log_termination_setting(logger, scenario)

    point_selector = RandomSelector()
    hp_seletor = HyperparameterizedSelector()
    tournament_dispatcher = MiniTournamentDispatcher()
    global_cache = TargetAlgorithmObserver.remote(scenario)
    if scenario.run_obj == "runtime":
        if scenario.monitor == "tournament_level":
            monitor = Monitor.remote(1, global_cache, scenario)
        elif scenario.monitor == "instance_level":
            monitor = InstanceMonitor.remote(1, global_cache, scenario)
        monitor.monitor.remote()

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
        #global_cache.put_tournament_history.remote(tournament)
        global_cache.put_tournament_update.remote(tournament)
        tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    #starting the monitor
    #global_cache.put_tournament_update.remote(tournaments)

    logger.info(f"Initial Tournaments {tournaments}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")

    main_loop_start = time.time()
    epoch = 0
    max_epochs = 256

    cutoff_time = scenario.cutoff_time
    predicted_quals = []
    predicted_perf = []
    evaluated = []
    qap = False

    # sm = SurrogateManager(scenario.parameter)
    fg = FeatureGenerator()
    sm = SurrogateManager(scenario, seed=42)
    smac_conf = sm.suggest(Surrogates.SMAC, scenario, n_samples=5)
    bug_handel = []
    tournament_history = {}

    while termination_check(scenario.termination_criterion, main_loop_start, scenario.wallclock_limit,
                            scenario.total_tournament_number, tournament_counter):

        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
            result_conf, result_instance, cancel_flag = result[0], result[1], result[2]

        # Some time a ray worker may crash. We handel that here. I.e if the TA did not run to the end, we reschedule
        except (ray.exceptions.WorkerCrashedError, ray.exceptions.TaskCancelledError) as e:
            logger.info(f'Crashed TA worker, {time.ctime()}, {winner}, {e}')
            # Figure out which tournament conf. belongs to
            for t in tournaments:
                conf_instance = get_tasks(t.ray_object_store, winner)
                if len(conf_instance) != 0:
                    tournament_of_c_i = t
                    break

            conf = [conf for conf in tournament_of_c_i.configurations if conf.id == conf_instance[0][0]][0]
            instance = conf_instance[0][1]
            # We check if we have killed the conf and only messed up the termination of the process

            termination_check_c_i = ray.get(global_cache.get_termination_single.remote(conf.id , instance))
            if termination_check_c_i:
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


        # Getting the tournament of the first task id
        first_task = tasks[0]
        ob_t = get_get_tournament_membership_with_ray_id(first_task, tournaments)

        # Figure out if the tournament of the first task is stale. If so cancel the task and start dummy task.
        if len(ob_t.configurations) == 1:
            i_no_result = get_instances_no_results(results, ob_t.configurations[0].id, ob_t.instance_set)
            if len(i_no_result) == 1:
                termination = ray.get(global_cache.get_termination_single.remote(ob_t.configurations[0].id, i_no_result[0]))
                result = ray.get(global_cache.get_results_single.remote(ob_t.configurations[0].id, i_no_result[0]))
                if termination and result == False and [ob_t.configurations[0],i_no_result[0]] not in bug_handel:
                    logger.info(f"Stale tournament: {time.strftime('%X %x %Z')}, {ob_t.configurations[0]}, {i_no_result[0]} , {first_task}, {bug_handel}")
                    ready_ids, _remaining_ids = ray.wait([first_task], timeout=0)
                    if len(_remaining_ids) == 1:
                        ray.cancel(first_task)
                        tasks.remove(first_task)
                        task = dummy_task.remote(ob_t.configurations[0],i_no_result[0], global_cache)
                        tasks.append(task)
                        bug_handel.append([ob_t.configurations[0],i_no_result[0]])


        #results = ray.get(global_cache.get_results.remote())
        if result_conf.id in list(results.keys()):
            results[result_conf.id][result_instance] = ray.get(global_cache.get_results_single.remote(result_conf.id,result_instance ))
        else:
            results[result_conf.id]= {}
            results[result_conf.id][result_instance] = ray.get(global_cache.get_results_single.remote(result_conf.id,result_instance ))


        result_tournament = get_tournament_membership(tournaments, result_conf)

        # Check whether we canceled a task or if the TA terminated regularly
        # In case we canceled a task, we need to remove it from the ray tasks
        if cancel_flag:
            if result_conf.id in result_tournament.ray_object_store.keys():
                if result_instance in result_tournament.ray_object_store[result_conf.id ].keys():
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

        logger.info(f"Result tournament update: {result_tournament}")

        if tournament_stop:
            print("Iteration:", time.time() - main_loop_start, tournament_counter)
            tournament_counter += 1

            # Get the instances for the new tournament
            instance_id, instances = instance_selector.get_subset(result_tournament.instance_set_id + 1)

            all_configs = result_tournament.best_finisher + result_tournament.worst_finisher

            terminations = ray.get(global_cache.get_termination_history.remote())

            for conf in all_configs:
                for surrogate in sm.surrogates.keys():
                    sm.update_surr(surrogate, result_tournament, all_configs, results, terminations)

            # Generate and select

            #generated_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            #points_to_run = point_selector.select_points(generated_points, scenario.tournament_size - 1, tournament_counter)

            random_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            # points_to_run = point_selector.select_points(generated_points, scenario.tournament_size-1, tournament_counter)
            default_ps = [default_point_generator.point_generator()]

            hist = {**tournament_history , **{t.id : t for t in tournaments}}
             #   ray.get(global_cache.get_tournament_history.remote())

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

            smac_conf = sm.suggest(Surrogates.SMAC, scenario, n_samples=5)

            generated_points = random_points + default_ps + \
                vg_points + lhc_ps + smac_conf

            features = fg.static_feature_gen(generated_points, epoch, max_epochs)
            features = np.concatenate(
                (features, fg.diversity_feature_gen(generated_points, hist,
                                                    results, cutoff_time,
                                                    scenario.parameter,
                                                    predicted_quals,
                                                    evaluated)),
                axis=1)

            for surrogate in sm.surrogates.keys():
                if sm.surrogates[surrogate].surr.model.rf is not None:
                    predicted_perf = sm.predict(surrogate,
                                                generated_points,
                                                cutoff_time)

            features = np.concatenate((features,
                                      fg.dynamic_feature_gen(generated_points,
                                                             hist,
                                                             predicted_perf,
                                                             sm, cutoff_time,
                                                             results)),
                                      axis=1)

            weights = [1 for _ in generated_points]
            weights = [weights for _ in features]
            weights = np.array(weights)

            points_to_run = hp_seletor.select_points(scenario, generated_points,
                                                     scenario.tournament_size - 1,
                                                     epoch, max_epochs, features, weights, results,
                                                     max_evals=100)

            for surrogate in sm.surrogates.keys():
                if sm.surrogates[surrogate].surr.model.rf is not None:
                    if qap:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          points_to_run,
                                                          cutoff_time))
                    else:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          evaluated,
                                                          cutoff_time))
                        qap = True

            evaluated.extend(points_to_run)

            #res = ray.get(global_cache.get_results.remote())

            points_to_run = points_to_run + [result_tournament.best_finisher[0]]


            # Create new tournament
            new_tournament, initial_assignments_new_tournament = tournament_dispatcher.init_tournament(results,
                                                                                                       points_to_run,
                                                                                                       instances,
                                                                                                       instance_id)
            # Remove that old tournament
            tournaments.remove(result_tournament)
            tournament_history[result_tournament.id] = result_tournament
            global_cache.put_tournament_history.remote(result_tournament)

            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            tournaments.append(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache,  ta_wrapper, scenario)

            global_cache.put_tournament_update.remote(new_tournament)
            global_cache.remove_tournament.remote(result_tournament)
           # global_cache.put_tournament_update.remote(tournaments)

            logger.info(f"Final results tournament {result_tournament}")
            logger.info(f"New tournament {new_tournament}")
            epoch += 1
            overall_best_update(tournaments, results, scenario)
        else:
            # If the tournament does not terminate we get a new conf/instance assignment and add that as ray task
            next_task = tournament_dispatcher.next_tournament_run(results, result_tournament, result_conf)
            tasks = update_tasks(tasks, next_task, result_tournament, global_cache, ta_wrapper, scenario)
            logger.info(f"New Task {next_task}, {result_tournament}")
            #global_cache.put_tournament_update.remote(tournaments)
            global_cache.put_tournament_update.remote(result_tournament)


    global_cache.save_rt_results.remote()
    global_cache.save_tournament_history.remote()

    print("DONE")
    logger.info("DONE")
    time.sleep(30)
    [ray.cancel(t) for t in not_ready]


if __name__ == "__main__":
    np.random.seed(42)

    parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament": 1, #import_wrapper
              "initial_instance_set_size": 2, "tournament_size": 2, "number_tournaments": 2, "total_tournament_number": 2,
              "total_runtime": 1200, "generator_multiple": 3, "set_size": 50,
              "termination_criterion": "total_tournament_number", "par": 1, "ta_pid_name": "glucose-simp", "memory_limit":1023*3, "log_folder":"run_1"}

    scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)#my_glucose_example #my_cadical_example

    check_log_folder(scenario.log_folder)
    clear_logs(scenario.log_folder)

    logging.basicConfig( level=logging.INFO,
                        format='%(asctime)s %(message)s', handlers = [
        logging.FileHandler(f"./selector/logs/{scenario.log_folder}/main.log"),
    ])

    logger = logging.getLogger(__name__)

    logger.info(f"Logging to {scenario.log_folder}")

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

    save_latest_logs(scenario.log_folder)
    ray.shutdown()






