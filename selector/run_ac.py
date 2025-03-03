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
    Run selector as a Python function.

    Parameters
    ----------
    scen_files : dict
        Paths to 'paramfile', 'instance_file', 'feature_file'.
    ray_mode : str
        'desktop' or 'cluster' (SLURM).
    kwargs : dict
        Anything else you want to set, see scenario.py.
    """

    # Add args to scenario
    for key, val in kwargs.items():
        if val != '':
            sys.argv.extend(['--' + key, str(val)])
        else:
            sys.argv.extend(['--' + key])

    selector_args = parse_args()
    selector_args['scenario_file'] = scen_files

    # Get wrapper for TA
    wrapper_mod = importlib.import_module(selector_args["wrapper_mod_name"])

    wrapper_name = selector_args["wrapper_class_name"]
    wrapper_ = getattr(wrapper_mod, wrapper_name)
    ta_wrapper = wrapper_()

    # Initialize scenario
    scenario = Scenario(selector_args["scenario_file"], selector_args)

    np.random.seed(scenario.seed)

    check_log_folder(scenario, scenario.log_folder)
    clear_logs(scenario, scenario.log_folder)

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if scenario.verbosity == 0:
        # Disable propagation to the root logger
        logger.propagate = False

        # Clear existing handlers (if any)
        if logger.hasHandlers():
            logger.handlers.clear()

    handler = \
        logging.FileHandler(
            f"{scenario.log_location}{scenario.log_folder}/main.log")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.info(f"Logging to {scenario.log_folder}")

    # Set up ray for desktop or HPC
    if ray_mode == 'desktop':
        ray.init()
    if ray_mode == 'cluster':
        ray.init(address="auto")

    logger.info("Ray info: {}".format(ray.cluster_resources()))
    logger.info("Ray nodes {}".format(ray.nodes()))
    logger.info("WD: {}".format(os.getcwd()))

    # Run AC process
    offline_mini_tournament_configuration(scenario, ta_wrapper, logger)

    print('\n')
    print('Processing results...')
    print('\n')

    # Process results
    save_latest_logs(scenario.log_folder, scenario)
    if scenario.run_obj == 'runtime':
        best = safe_best(f'./selector/logs/{scenario.log_folder}/',
                         scenario.cutoff_time)
    elif scenario.run_obj == 'quality':
        best = safe_best(f'./selector/logs/{scenario.log_folder}/',
                         scenario.crash_cost)

    print('Best Configuration:\n', best[list(best.keys())[0]]['conf'])
    ray.shutdown()

    print('\n')
    print(f'See ./selector/logs/{scenario.log_folder}/')
    print('\n')


if __name__ == "__main__":
    pass
