import unittest
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
import selector.tournament_performance as tp


class TestTournamentPerformance(unittest.TestCase):

    def setUp(self):
        self.instances_set = ["instance_" + f"{n}" for n in range(1, 11)]
        self.configuration_id = "configuration_1"
        self.results = {self.configuration_id: {}}
        self.runtime = 0
        for instance in self.instances_set:
            self.results[self.configuration_id][instance] = np.random.uniform(0, 5)
            self.runtime += self.results[self.configuration_id][instance]

    def test_get_conf_time_out(self):
        conf_to = tp.get_conf_time_out(self.results, self.configuration_id, self.instances_set)
        self.assertEqual(conf_to, False)

        conf_to = tp.get_conf_time_out(self.results, "configuration_2", self.instances_set)
        self.assertEqual(conf_to, None)

        self.results[self.configuration_id][self.instances_set[0]] = np.nan
        conf_to = tp.get_conf_time_out(self.results, self.configuration_id, self.instances_set)
        self.assertEqual(conf_to, True)

    def test_get_censored_runtime_for_instance_set(self):
        censored_runtime = tp.get_censored_runtime_for_instance_set(self.results, self.configuration_id,
                                                                    self.instances_set)
        self.assertEqual(round(censored_runtime, 10), round(self.runtime, 10))

        instance_subset = np.random.choice(self.instances_set, 5, replace=False).tolist()
        runtime_subset = 0
        for instance in instance_subset:
            runtime_subset += self.results[self.configuration_id][instance]
        censored_runtime = tp.get_censored_runtime_for_instance_set(self.results, self.configuration_id,
                                                                    instance_subset)
        self.assertEqual(round(censored_runtime, 10), round(runtime_subset, 10))

        runtime_subset -= self.results[self.configuration_id][instance_subset[0]]
        self.results[self.configuration_id][instance_subset[0]] = np.nan
        censored_runtime = tp.get_censored_runtime_for_instance_set(self.results, self.configuration_id,
                                                                    instance_subset)
        self.assertEqual(round(censored_runtime, 10), round(runtime_subset, 10))

    def test_get_runtime_for_instance_set_with_timeout(self):
        timeout = 2
        par_penalty = 10
        runtime_to = tp.get_runtime_for_instance_set_with_timeout(self.results, self.configuration_id,
                                                                  self.instances_set, timeout, par_penalty)
        self.assertEqual(round(runtime_to, 10), round(self.runtime, 10))

        self.runtime -= self.results[self.configuration_id][self.instances_set[0]]
        self.runtime += timeout * par_penalty
        self.results[self.configuration_id][self.instances_set[0]] = np.nan
        runtime_to = tp.get_runtime_for_instance_set_with_timeout(self.results, self.configuration_id,
                                                                  self.instances_set, timeout, par_penalty)
        self.assertEqual(round(runtime_to, 10), round(self.runtime, 10))

    def test_get_censored_runtime_of_configuration(self):
        censored_runtime = tp.get_censored_runtime_of_configuration(self.results, self.configuration_id)
        self.assertEqual(round(censored_runtime, 10), round(self.runtime, 10))

        self.runtime -= self.results[self.configuration_id][self.instances_set[0]]
        self.results[self.configuration_id][self.instances_set[0]] = np.nan
        censored_runtime = tp.get_censored_runtime_of_configuration(self.results, self.configuration_id)
        self.assertEqual(round(censored_runtime, 10), round(self.runtime, 10))

    def test_get_instances_no_results(self):
        instances_not_run_on = ["instance_" + f"{n}" for n in range(11, 16)]
        self.instances_set += instances_not_run_on
        not_run_on = tp.get_instances_no_results(self.results, self.configuration_id, self.instances_set)
        self.assertEqual(not_run_on, instances_not_run_on)


if __name__ == "__main__":
    unittest.main()
