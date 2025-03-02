import unittest
import ray
import sys
import os
import numpy as np
import time
import copy
sys.path.append(os.getcwd())
from selector.test.tap_work_wrapper import TAP_Work_Wrapper
from selector.scenario import Scenario
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.pointselector import RandomSelector
from selector.instance_sets import InstanceSet
from selector.tournament_dispatcher import MiniTournamentDispatcher
from selector.ta_result_store import TargetAlgorithmObserver
import selector.tournament_bookkeeping as tb


class TestTournamentBookkeeping(unittest.TestCase):

    def setUp(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        print(dir_path)
        print(dir_path)
        print(dir_path)
        print(dir_path)
        print(dir_path)
        self.parser = {"check_path": False, "seed": 42, "ta_run_type": "import_wrapper", "winners_per_tournament": 1,
                  # import_wrapper
                  "initial_instance_set_sideterministicze": 2, "tournament_size": 3, "number_tournaments": 3,
                  "total_tournament_number": 2, "cpu_binding": False, 'instances_dir': './test/test_data/instances',
                  "total_runtime": 1200, "generator_multiple": 3, "set_size": 50, "solve_match": [],
                  "termination_criterion": "total_runtime", "par": 1, "ta_pid_name": "glucose-simp",
                  "memory_limit": 1023 * 3, "log_folder": "run_1", "initial_instance_set_size": 5, "runtime_feedback": ""}
        self.scenario = Scenario("./test_data/test_example.txt", self.parser)
        self.ta_wrapper = TAP_Work_Wrapper()
        self.global_cache = TargetAlgorithmObserver.remote(self.scenario)
        self.random_generator = PointGen(self.scenario, random_point)
        self.point_selector = RandomSelector()
        self.instance_selector = InstanceSet(self.scenario.instance_set, self.scenario.initial_instance_set_size,
                                             self.scenario.set_size)
        self.tournament_dispatcher = MiniTournamentDispatcher()
        self.tournament_counter = 0
        self.tasks = []
        self.tournaments = []
        self.results = ray.get(self.global_cache.get_results.remote())

        self.initial_assignments = []

        for _ in range(self.scenario.number_tournaments):
            generated_points = [self.random_generator.point_generator() for _ in range(self.scenario.tournament_size *
                                                                                  self.scenario.generator_multiple)]

            points_to_run = self.point_selector.select_points(generated_points, self.scenario.tournament_size,
                                                         self.tournament_counter)

            instance_id, instances = self.instance_selector.get_subset(0)
            tournament, initial_assignments = self.tournament_dispatcher.init_tournament(self.results, points_to_run,
                                                                                         instances, instance_id)
            self.initial_assignments.append(initial_assignments)
            self.tournaments.append(tournament)
            self.global_cache.put_tournament_history.remote(tournament)
            self.global_cache.put_tournament_update.remote(tournament)
            self.tasks = tb.update_tasks(self.tasks, initial_assignments, tournament, self.global_cache,
                                         self.ta_wrapper, self.scenario)

    def test_get_tournament_membership(self):
        conf_ids = []
        for t in self.tournaments:
            for conf in t.configurations:
                conf_ids.append(conf)
        test_conf = np.random.choice(conf_ids)

        t = tb.get_tournament_membership(self.tournaments, test_conf)

        self.assertIn(test_conf, t.configurations)

    def test_get_get_tournament_membership_with_ray_id(self):

        test_task = np.random.choice(self.tasks, 1)[0]
        tournament = tb.get_get_tournament_membership_with_ray_id(test_task, self.tournaments)

        tournament_tasks = []
        for conf in tournament.ray_object_store:
            for instance in tournament.ray_object_store[conf]:
                tournament_tasks.append(tournament.ray_object_store[conf][instance])

        self.assertIn(test_task, tournament_tasks)

    def test_get_tasks(self):

        for i in range(len(self.tournaments)):
            assignments = []
            for assignment in self.initial_assignments[i]:
                assignments.append([assignment[0].id, assignment[1]])
            tasks = tb.get_tasks(self.tournaments[i].ray_object_store, self.tasks)
            self.assertEqual(assignments, tasks)

    def test_update_tasks(self):

        original_tasks = copy.deepcopy(self.tasks)

        generated_points = [self.random_generator.point_generator() for _ in range(self.scenario.tournament_size *
                                                                                   self.scenario.generator_multiple)]

        points_to_run = self.point_selector.select_points(generated_points, self.scenario.tournament_size,
                                                          self.tournament_counter)

        instance_id, instances = self.instance_selector.get_subset(0)
        tournament, initial_assignments = self.tournament_dispatcher.init_tournament(self.results, points_to_run,
                                                                                     instances, instance_id)
        self.tournaments.append(tournament)
        self.global_cache.put_tournament_history.remote(tournament)
        self.global_cache.put_tournament_update.remote(tournament)
        self.tasks = tb.update_tasks(self.tasks, initial_assignments, tournament, self.global_cache, self.ta_wrapper,
                                     self.scenario)

        self.assertTrue(len(original_tasks) + self.scenario.tournament_size == len(self.tasks))

        new_assignment = []
        for assignment in initial_assignments:
            new_assignment.append([assignment[0].id, assignment[1]])
        new_tasks = tb.get_tasks(tournament.ray_object_store, self.tasks)

        self.assertEqual(new_assignment, new_tasks)

    def test_termination_check(self):
        start_time = time.time()
        termination = tb.termination_check(termination_criterion="total_runtime", main_loop_start=start_time,
                                           total_runtime=10, total_tournament_number=1, tournament_counter=1)
        self.assertTrue(termination)

        time.sleep(1)
        termination = tb.termination_check(termination_criterion="total_runtime", main_loop_start=start_time,
                                           total_runtime=1, total_tournament_number=1, tournament_counter=1)
        self.assertFalse(termination)

        termination = tb.termination_check(termination_criterion="total_tournament_number", main_loop_start=start_time,
                                           total_runtime=1, total_tournament_number=1, tournament_counter=0)
        self.assertTrue(termination)

        termination = tb.termination_check(termination_criterion="total_tournament_number", main_loop_start=start_time,
                                           total_runtime=1, total_tournament_number=1, tournament_counter=1)
        self.assertFalse(termination)

        start_time = time.time()
        termination = tb.termination_check(termination_criterion="not defined", main_loop_start=start_time,
                                           total_runtime=10, total_tournament_number=1, tournament_counter=1)
        self.assertTrue(termination)

        time.sleep(1)
        termination = tb.termination_check(termination_criterion="not defined", main_loop_start=start_time,
                                           total_runtime=1, total_tournament_number=1, tournament_counter=1)
        self.assertFalse(termination)


if __name__ == "__main__":
    ray.init()
    unittest.main()



