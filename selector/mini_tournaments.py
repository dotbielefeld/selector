"""
In this module the main while loop for tournaments, configuration generation
and feedback is defined.
"""
import logging
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import ray
import time
from sklearn_extra.cluster import KMedoids


from selector.scenario import Scenario
from selector.pointselector import HyperparameterizedSelector
from selector.ta_result_store import TargetAlgorithmObserver
from selector.ta_execution import dummy_task

from selector.point_gen import PointGen
from selector.pool import Surrogates
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import default_point
from selector.generators.variable_graph_point_generator import variable_graph_point, Mode
from selector.generators.lhs_point_generator import lhc_points, LHSType, Criterion
from selector.selection_features import FeatureGenerator
from selector.generators.surrogates.surrogates import SurrogateManager
# from selector.surrogates.surrogates import SurrogateManager

from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.tournament_bookkeeping import get_tournament_membership, update_tasks, get_tasks, termination_check, get_get_tournament_membership_with_ray_id
from selector.log_setup import clear_logs, log_termination_setting, check_log_folder, save_latest_logs

from selector.tournament_monitor import Monitor
from selector.tournament_performance import overall_best_update, get_instances_no_results

# from selector.wrapper.tap_work_wrapper import TAP_Work_Wrapper
from selector.instance_sets import TimedInstanceSet
from selector.instance_monitor import InstanceMonitor
from selector.best_conf import safe_best
from selector.cleanup import TempFileCleaner


def offline_mini_tournament_configuration(scenario, ta_wrapper, logger):
    """
    Manages the tournaments, suggestions, selection and feedback.

    Parameters
    ----------
    scenario : selector.scenario.Scenario
        AS scenario.
    ta_wrapper : UserWrapperClass
        Wrapper that generates command lines to call target algorithm.
    logger : logging.Logger
        Logging object.
    """
    log_termination_setting(logger, scenario)

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

    instance_selector = TimedInstanceSet(scenario.instance_set, scenario.initial_instance_set_size, scenario.set_size, runtime=scenario.wallclock_limit, start_time=0.15, end_time=0.7)
    tasks = []
    tournaments = []
    tournament_counter = 0
    results = ray.get(global_cache.get_results.remote())

    # creating the first tournaments and adding first conf/instance pairs to ray tasks
    for tc in range(scenario.number_tournaments):
        if tc == 0:
            points_to_run = [random_generator.point_generator() for _ in range(scenario.tournament_size-1)] + [default_point_generator.point_generator()]
        else:
            points_to_run = [random_generator.point_generator() for _ in range(scenario.tournament_size)]

        instance_id, instances = instance_selector.get_subset(0, 0, 0)
        tournament, initial_assignments = tournament_dispatcher.init_tournament(results, points_to_run,
                                                                                instances, instance_id)
        tournaments.append(tournament)
        #global_cache.put_tournament_history.remote(tournament)
        global_cache.put_tournament_update.remote(tournament)
        tasks = update_tasks(tasks, initial_assignments, tournament, global_cache, ta_wrapper, scenario)

    #starting the monitor

    logger.info(f"Initial Tournaments {tournaments}")
    logger.info(f"Initial Tasks, {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")

    if scenario.cleanup:
        cleaner = TempFileCleaner(logger, age_limit=scenario.cutoff_time * 2)

    main_loop_start = time.time()
    epoch = 0
    max_epochs = 256

    inject_default = 0

    cutoff_time = scenario.cutoff_time
    predicted_quals = []
    evaluated = []
    qap = False

    fg = FeatureGenerator(logger=None)
    sm = SurrogateManager(scenario, logger=logger)

    bug_handel = []
    tournament_history = {}
    surrogate_amortized_time = 20
    next_surrogate_update = 1
    surrogate_update_counter = 1

    while termination_check(scenario.termination_criterion, main_loop_start, scenario.wallclock_limit,
                            scenario.total_tournament_number, tournament_counter):

        winner, not_ready = ray.wait(tasks)
        tasks = not_ready
        try:
            result = ray.get(winner)[0]
            result_conf, result_instance, cancel_flag = result[0], result[1], result[2]

        # Some time a ray worker may crash. We handel that here. I.e if the TA did not run to the end, we reschedule
        except (ray.exceptions.WorkerCrashedError, ray.exceptions.TaskCancelledError, ray.exceptions.RayTaskError, TypeError) as e:
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
            else:  # got no results: need to rescheulde
                next_task = [[conf, instance]]
                tasks = update_tasks(tasks, next_task, tournament_of_c_i, global_cache, ta_wrapper, scenario)
                logger.info(f"There are no results: rescheduling {conf.id}, {instance} {[get_tasks(o.ray_object_store, tasks) for o in tournaments]}")
                continue

        # Getting the tournament of the first task id
        if len(tasks) > 0:
            first_task = tasks[0]
            ob_t = get_get_tournament_membership_with_ray_id(first_task, tournaments)

            # Figure out if the tournament of the first task is stale. If so cancel the task and start dummy task.
            if ob_t is not None:
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
        logger.info(f"\nTournament result: {result_tournament}")

        if tournament_stop:
            tournament_counter += 1

            if scenario.cleanup:
                cleaner.clean_up()

            # Get the instances for the new tournament
            instance_id, instances = instance_selector.get_subset(result_tournament.instance_set_id + 1, time.time() - main_loop_start, result_tournament.instance_set_id + 1)
            all_configs = result_tournament.best_finisher + result_tournament.worst_finisher

            terminations = ray.get(global_cache.get_termination_history.remote())

            ac_runtime = time.time() - main_loop_start

            overall_best_update(tournaments, results, scenario, ac_runtime)

            # Remove that old tournament
            tournaments.remove(result_tournament)
            tournament_history[result_tournament.id] = result_tournament
            global_cache.put_tournament_history.remote(result_tournament)

            logger.info(f"Results on instances: {results}")
            for surrogate in sm.surrogates.keys():
                start_update = time.time()
                if surrogate == Surrogates.GGApp:
                    if surrogate_update_counter == next_surrogate_update:
                        sm.update_surr(surrogate, result_tournament, all_configs, results, terminations, ac_runtime)

                        surrogate_update_counter = 0
                elif surrogate == Surrogates.SMAC:
                    sm.update_surr(surrogate, result_tournament, all_configs, results, terminations, ac_runtime)
                else:
                    sm.update_surr(surrogate, result_tournament, all_configs, results, terminations)
                surrogate_time = time.time() - start_update

            surrogate_update_counter = surrogate_update_counter + 1

            if surrogate_time >= surrogate_amortized_time:
                next_surrogate_update = next_surrogate_update + scenario.model_update_iteration
                surrogate_amortized_time = surrogate_amortized_time + surrogate_amortized_time

            # Generate and select
            random_points = [random_generator.point_generator() for _ in range(scenario.tournament_size * scenario.generator_multiple)]
            default_ps = [default_point_generator.point_generator()]

            hist = {**tournament_history, **{t.id: t for t in tournaments}}

            vg_points = [vg_point_generator.point_generator(
                         results=results, mode=Mode.random, alldata=hist,
                         lookback=1)
                         for _ in range(
                         scenario.tournament_size *
                         scenario.generator_multiple)]

            lhc_ps = lhc_point_generator.point_generator(
                n_samples=(scenario.tournament_size *
                           scenario.generator_multiple),
                lhs_type=LHSType.centered,
                criterion=Criterion.maximin)

            smac_conf = \
                sm.suggest(Surrogates.SMAC, scenario,
                           scenario.tournament_size * scenario.generator_multiple,
                           None, None, instances)

            ggapp_conf = \
                sm.suggest(Surrogates.GGApp, scenario,
                           scenario.tournament_size * scenario.generator_multiple,
                           hist, results, None)

            cppl_conf = \
                sm.suggest(Surrogates.CPPL, scenario,
                           scenario.tournament_size * scenario.generator_multiple,
                           None, None, instances)[0]
            
            generated_points = random_points + default_ps + \
                vg_points + lhc_ps + smac_conf + ggapp_conf + cppl_conf

            feature_time_start = time.time()

            features = \
                fg.static_feature_gen(generated_points, epoch, max_epochs)
            features = np.concatenate(
                (features, fg.diversity_feature_gen(generated_points, hist,
                                                    results, cutoff_time,
                                                    scenario.parameter,
                                                    predicted_quals,
                                                    evaluated)),
                axis=1)

            features = np.concatenate((features,
                                      fg.dynamic_feature_gen(generated_points,
                                                             hist,
                                                             None,
                                                             sm, cutoff_time,
                                                             results,
                                                             instances)),
                                      axis=1)
        
            feature_time = time.time() - feature_time_start

            set_weights = [value for hp, value in scenario.__dict__.items()
                           if hp[:2] == 'w_']
            weights = [set_weights for _ in generated_points]
            weights = np.array(weights)

            selection_start = time.time()

            points_to_run = \
                hp_seletor.select_points(scenario, generated_points,
                                         scenario.tournament_size - 1,
                                         epoch, max_epochs, features, weights,
                                         results, max_evals=100)

            select_time = time.time() - selection_start

            logger.info(f"All points generated \n\n{generated_points}\n\n")
            logger.info(f"Features computed \n\n{features.tolist()}\n\n")
            logger.info(f"Points selected to run \n\n{points_to_run}\n\n")
            logger.info(f"Instance set \n\n{instances}\n\n")
            logger.info(f"Terminations \n\n{terminations}\n\n")

            if scenario.wallclock_limit * inject_default < time.time() - main_loop_start:
                points_to_run[-1] = default_point_generator.point_generator()
                if tournament_counter % scenario.number_tournaments == 0:
                    inject_default += 0.25

            points_to_run = points_to_run + [result_tournament.best_finisher[0]]

            evaluated.extend(points_to_run)
            # Reduce evaluated list to number_tournaments*tournament_size
            # after tn tournaments
            if tournament_counter % scenario.tn == 0 and \
                    len(evaluated) > len(points_to_run):
                eval_np = []
                for ev_conf in evaluated:
                    eval_np.append(
                        sm.surrogates[Surrogates.GGApp].
                        transform_values(ev_conf))
                eval_np = np.asarray(eval_np)
                kmedoids = \
                    KMedoids(
                        n_clusters=scenario.number_tournaments * scenario.
                        tournament_size, random_state=0).fit(eval_np)
                clusters = kmedoids.cluster_centers_
                new_evaluated = []
                clusters = [c.tolist() for c in clusters]
                eval_np = [e.tolist() for e in eval_np]
                for c in clusters:
                    new_evaluated.append(evaluated[eval_np.index(c)])
                evaluated = new_evaluated

            pred_feature_time_start = time.time()

            for surrogate in sm.surrogates.keys():
                if surrogate is Surrogates.SMAC:
                    if sm.surrogates[surrogate].surr.model.rf is not None:
                        if qap:
                            predicted_quals.extend(sm.predict(surrogate,
                                                              points_to_run,
                                                              cutoff_time,
                                                              instances))
                        elif len(evaluated) != 0:
                            predicted_quals.extend(sm.predict(surrogate,
                                                              evaluated,
                                                              cutoff_time,
                                                              instances))
                            qap = True

                else:
                    if qap:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          points_to_run,
                                                          cutoff_time,
                                                          instances))
                    elif len(evaluated) != 0:
                        predicted_quals.extend(sm.predict(surrogate,
                                                          evaluated,
                                                          cutoff_time,
                                                          instances))
                        qap = True

            pred_feat_time = time.time() - pred_feature_time_start

            # Create new tournament
            new_tournament, initial_assignments_new_tournament = tournament_dispatcher.init_tournament(results,
                                                                                                       points_to_run,
                                                                                                       instances,
                                                                                                       instance_id)

            # Add the new tournament and update the ray tasks with the new conf/instance assignments
            tournaments.append(new_tournament)
            tasks = update_tasks(tasks, initial_assignments_new_tournament, new_tournament, global_cache,  ta_wrapper, scenario)

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
            global_cache.put_tournament_update.remote(result_tournament)

    global_cache.save_rt_results.remote()
    global_cache.save_tournament_history.remote()

    logger.info("AC run completed!")
    time.sleep(30)
    [ray.cancel(t) for t in not_ready]


if __name__ == "__main__":
    np.random.seed(42)

    parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament": 1, #import_wrapper
              "initial_instance_set_size": 2, "tournament_size": 2, "number_tournaments": 2, "total_tournament_number": 2,
              "total_runtime": 1200, "generator_multiple": 3, "set_size": 50, "monitor": "tournament_level",
              "termination_criterion": "total_tournament_number", "par": 1, "ta_pid_name": "glucose-simp", "memory_limit":1023*3, "log_folder":"run_1",
              "wrapper_mod_name": "selector.wrapper.cadical_wrapper", "wrapper_class_name": "Cadical_Wrapper"}

    scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)#my_glucose_example #my_cadical_example

    check_log_folder(scenario, scenario.log_folder)
    clear_logs(scenario, scenario.log_folder)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(f"./selector/logs/{scenario.log_folder}/main.log"),
    ])

    logger = logging.getLogger(__name__)

    logger.info(f"Logging to {scenario.log_folder}")

    scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)
    # TODO this needs to come from the scenario?!
    ta_wrapper = TAP_Work_Wrapper()

    # init
    ray.init(address="auto")
    # ray.init()

    logger.info("Ray info: {}".format(ray.cluster_resources()))
    logger.info("Ray nodes {}".format(ray.nodes()))
    logger.info("WD: {}".format(os.getcwd()))

    offline_mini_tournament_configuration(scenario, ta_wrapper, logger)

    save_latest_logs(scenario.log_folder, scenario)
    safe_best(sys.path[-1] + f'/selector/logs/{scenario.log_folder}/',
              scenario.cutoff_time)
    ray.shutdown()
