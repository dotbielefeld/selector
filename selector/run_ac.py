"""Module of selector to run from Python."""

import sys
import os
sys.path.append(os.getcwd())

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import importlib
import logging
import numpy as np
import ray

from selector.scenario import Scenario, parse_args
from selector.log_setup import clear_logs, check_log_folder, save_latest_logs
from selector.mini_tournaments import offline_mini_tournament_configuration
from selector.best_conf import safe_best

sys.path.append(os.getcwd())

def ac(scen_files, ray_mode, **kwargs):
    """
    Run selector as a python function.

    :param scen_files: dict, paths to 'paramfile', 'instance_file', 'feature_file'
    :param ray_mode: str, 'desktop' or 'cluster'
    :param kwargs: Anything else you want to set, see scenario.py
    :return:
    """
    for key, val in kwargs.items():
        sys.argv.extend(['--' + key, str(val)])

    selector_args = parse_args()
    selector_args['scenario_file'] = scen_files

    wrapper_mod = importlib.import_module(selector_args["wrapper_mod_name"])

    wrapper_name = selector_args["wrapper_class_name"]
    wrapper_ = getattr(wrapper_mod, wrapper_name)
    ta_wrapper = wrapper_()

    scenario = Scenario(selector_args["scenario_file"], selector_args)

    np.random.seed(scenario.seed)

    check_log_folder(scenario.log_folder)
    clear_logs(scenario.log_folder)

    logging.\
        basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(
                        f"./selector/logs/{scenario.log_folder}/main.log"), ])

    logger = logging.getLogger(__name__)

    logger.info(f"Logging to {scenario.log_folder}")

    if ray_mode == 'desktop':
        ray.init()
    if ray_mode == 'cluster':
        ray.init(address="auto")

    logger.info("Ray info: {}".format(ray.cluster_resources()))
    logger.info("Ray nodes {}".format(ray.nodes()))
    logger.info("WD: {}".format(os.getcwd()))

    offline_mini_tournament_configuration(scenario, ta_wrapper, logger)

    print('\n')
    print('Processing results...')

    save_latest_logs(scenario.log_folder)
    if scenario.termination_criterion == 'runtime':
        safe_best(f'./selector/logs/{scenario.log_folder}/',
                  scenario.cutoff_time)
    elif scenario.termination_criterion == 'quality':
        safe_best(f'./selector/logs/{scenario.log_folder}/',
                  sys.maxsize)
    ray.shutdown()

    print('\n')
    print(f'See ./selector/logs/{scenario.log_folder}/')
    print('\n')

if __name__ == "__main__":
    pass
