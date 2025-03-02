import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from selector.scenario import Scenario
from selector.pool import ParamType


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        self.scenario_dict = {'ta_cmd': 'echo test',
                                   'paramfile': './test/test_data/test_params.pcs',
                                   'execdir': '.',
                                   'deterministic': 0,
                                   'run_obj': 'runtime',
                                   'overall_obj': 'mean10',
                                   'cutoff_time': 300,
                                   'wallclock_limit': 172800,
                                   'instance_file':
                                       './test/test_data/test_training.txt',
                                   'test_instance_file':
                                       './test/test_data/test_test.txt',
                                   'feature_file':
                                       './test/test_data/test_features.txt',
                                   'output_dir':
                                       'placeholder',
                                    'instances_dir': './test/test_data/instances'}

    def test_scenario_type(self):
        with self.assertRaises(TypeError):
            _ = Scenario(scenario=2)

    def test_scenario_from_sting(self):
        scenario = Scenario('./test/test_data/test_scenario.txt')

        self.assertEqual(scenario.ta_cmd, 'echo test')
        self.assertEqual(scenario.execdir, '.')
        self.assertEqual(scenario.deterministic, "0")
        self.assertEqual(scenario.run_obj, 'runtime')
        self.assertEqual(scenario.overall_obj, 'mean10')
        self.assertEqual(scenario.cutoff_time, 300.0)
        self.assertEqual(scenario.wallclock_limit, 172800)
        self.assertEqual(scenario.instance_file, './test/test_data/test_training.txt')
        self.assertEqual(scenario.test_instance_file, './test/test_data/test_test.txt')
        self.assertEqual(scenario.feature_file, './test/test_data/test_features.txt')
        self.assertEqual(scenario.paramfile,'./test/test_data/test_params.pcs')
        self.assertEqual(scenario.instance_set,['./test/test_data/instances/test_instance_1.cnf'])
        self.assertEqual(scenario.test_instances, ['./test/test_data/instances/test_instance_2.cnf'])
        self.assertEqual(scenario.feature_names, ['feature_1', ' feature_2'])

        features_1 = scenario.features['./test/test_data/instances/test_instance_1.cnf']
        self.assertTrue(np.allclose(features_1, [90., 7650.]))

        features_2 = scenario.features['./test/test_data/instances/test_instance_2.cnf']
        self.assertTrue(np.allclose(features_2, [30., 8045.1]))

    def test_scenario_from_dic(self):
        scenario = Scenario(self.scenario_dict)

        self.assertEqual(scenario.ta_cmd, 'echo test')
        self.assertEqual(scenario.execdir, '.')
        self.assertEqual(scenario.deterministic, 0)
        self.assertEqual(scenario.run_obj, 'runtime')
        self.assertEqual(scenario.overall_obj, 'mean10')
        self.assertEqual(scenario.cutoff_time, 300)
        self.assertEqual(scenario.wallclock_limit, 172800)
        self.assertEqual(scenario.instance_file,'./test/test_data/test_training.txt')
        self.assertEqual(scenario.test_instance_file,'./test/test_data/test_test.txt')
        self.assertEqual(scenario.feature_file,'./test/test_data/test_features.txt')
        self.assertEqual(scenario.paramfile,'./test/test_data/test_params.pcs')
        self.assertEqual(scenario.instance_set,['./test/test_data/instances/test_instance_1.cnf'])
        self.assertEqual(scenario.test_instances, ['./test/test_data/instances/test_instance_2.cnf'])
        self.assertEqual(scenario.feature_names, ['feature_1', ' feature_2'])

        features_1 = scenario.features['./test/test_data/instances/test_instance_1.cnf']
        self.assertTrue(np.allclose(features_1, [  90., 7650.]))

        features_2 = scenario.features['./test/test_data/instances/test_instance_2.cnf']
        self.assertTrue(np.allclose(features_2, [  30., 8045.1]))

    def test_parameter_from_file(self):
        scenario_file = Scenario('./test/test_data/test_scenario.txt')
        scenario_dic = Scenario(self.scenario_dict)

        for scenario in [scenario_file,scenario_dic ]:
            test_parameter_list = scenario.parameter

            test_no_goods = scenario.no_goods

            # test luby
            self.assertEqual(test_parameter_list[0].name, 'luby')
            self.assertEqual(test_parameter_list[0].type, ParamType.categorical)
            self.assertEqual(test_parameter_list[0].bound, [False, True])
            self.assertEqual(test_parameter_list[0].default, False)
            self.assertEqual(test_parameter_list[0].condition, {})
            self.assertEqual(test_parameter_list[0].scale, '')

            # test rinc
            self.assertEqual(test_parameter_list[1].name, 'rinc')
            self.assertEqual(test_parameter_list[1].type, ParamType.continuous)
            self.assertEqual(test_parameter_list[1].bound, [1.1, 4.0])
            self.assertEqual(test_parameter_list[1].default, 2)
            self.assertEqual(test_parameter_list[1].condition, {'luby': [True, False]})
            self.assertEqual(test_parameter_list[1].scale, '')

            # test cla-decay
            self.assertEqual(test_parameter_list[2].name, 'cla-decay')
            self.assertEqual(test_parameter_list[2].type, ParamType.continuous)
            self.assertEqual(test_parameter_list[2].bound, [0.9, 0.99999])
            self.assertEqual(test_parameter_list[2].default, 0.999)
            self.assertEqual(test_parameter_list[2].condition, {})
            self.assertEqual(test_parameter_list[2].scale, 'l')

            # test phase-saving
            self.assertEqual(test_parameter_list[3].name, 'phase-saving')
            self.assertEqual(test_parameter_list[3].type, ParamType.integer)
            self.assertEqual(test_parameter_list[3].bound, [0, 2])
            self.assertEqual(test_parameter_list[3].default, 2)
            self.assertEqual(test_parameter_list[3].condition, {})
            self.assertEqual(test_parameter_list[3].scale, '')

            # test strSseconds
            self.assertEqual(test_parameter_list[4].name, 'strSseconds')
            self.assertEqual(test_parameter_list[4].type, ParamType.categorical)
            self.assertEqual(test_parameter_list[4].bound, ['10', '50', '100', '150', '200', '250', '290'])
            self.assertEqual(test_parameter_list[4].default, '150')
            self.assertEqual(test_parameter_list[4].condition, {'luby': [True], 'cla-decay': [0.92, 0.93]})
            self.assertEqual(test_parameter_list[4].scale, '')

            # test bce-limit
            self.assertEqual(test_parameter_list[5].name, 'bce-limit')
            self.assertEqual(test_parameter_list[5].type, ParamType.integer)
            self.assertEqual(test_parameter_list[5].bound,[100000,200000000])
            self.assertEqual(test_parameter_list[5].default, 100000000)
            self.assertEqual(test_parameter_list[5].condition, {})
            self.assertEqual(test_parameter_list[5].scale, 'l')

            # test no goods
            self.assertEqual(test_no_goods[0], {"luby": True , "rinc": 3})

    def test_instance_file_avail(self):
        with self.assertRaises(FileExistsError):
            _ = Scenario(self.scenario_dict, cmd={'check_path': True})


if __name__ == "__main__":
    unittest.main()

