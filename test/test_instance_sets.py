import unittest
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from selector.scenario import Scenario
from selector.instance_sets import InstanceSet


class TestInstanceSets(unittest.TestCase):

    def setUp(self):
        sys.path.append(os.getcwd())
        parser = {"check_path": False, "initial_instance_set_size": 3, "set_size": 6}
        self.scenario = Scenario("./selector/input/scenarios/test_example.txt", parser)
        self.instance_selector = InstanceSet(self.scenario.instance_set, self.scenario.initial_instance_set_size,
                                             self.scenario.set_size)

    def test_InstanceSet(self):
        self.assertEqual(self.instance_selector.instance_set, self.scenario.instance_set)
        self.assertEqual(self.instance_selector.start_instance_size, self.scenario.initial_instance_set_size)
        self.assertEqual(self.instance_selector.set_size, self.scenario.set_size)
        self.assertEqual(self.instance_selector.instance_sets, [])
        self.assertEqual(self.instance_selector.subset_counter, 0)
        self.assertEqual(self.instance_selector.instance_increment_size,
                         round(len(self.instance_selector.instance_set) /
                               np.floor(len(self.instance_selector.instance_set) /
                                        self.instance_selector.start_instance_size), 1))

    def test_next_set(self):
        with self.assertRaises(ValueError):
            self.instance_selector.instance_set = []
            self.instance_selector.next_set()

        self.instance_selector.instance_set = self.scenario.instance_set
        self.instance_selector.next_set()
        self.assertEqual(self.instance_selector.subset_counter, 1)
        self.assertIsNot(self.instance_selector.instance_sets, [])

        self.instance_selector.next_set()
        self.instance_selector.next_set()
        self.assertEqual(self.instance_selector.instance_sets[-1], self.instance_selector.instance_sets[-2])

    def test_get_subset(self):
        with self.assertRaises(AssertionError):
            self.instance_selector.get_subset(1)

        next_tournament_set_id, next_set = self.instance_selector.get_subset(0)
        self.assertEqual(next_tournament_set_id, 0)
        self.assertEqual(len(next_set), int(self.instance_selector.instance_increment_size))

        next_tournament_set_id_2, next_set_2 = self.instance_selector.get_subset(self.instance_selector.subset_counter)
        self.assertEqual(len(next_set_2), len(next_set) + int(self.instance_selector.instance_increment_size))

        next_tournament_set_id_3, next_set_3 = self.instance_selector.get_subset(0)
        self.assertEqual(next_set, next_set_3)

        next_tournament_set_id_4, next_set_4 = self.instance_selector.get_subset(self.instance_selector.subset_counter)
        self.assertEqual(len(next_set_4), len(next_set_2))


if __name__ == "__main__":
    unittest.main()
