"""In this module logging is set up."""
import os
import shutil
from datetime import datetime
import dataclasses
import json
import uuid
from enum import Enum
import ray
import numpy as np


def clear_logs(scenario, folder_for_run=None):
    """
    Clear the logs.

    Parameters
    ----------
    folder_for_run : str
        Path to log directory.
    """
    if folder_for_run is None:
        folder_for_run = "latest"

    for folder in [f'{scenario.log_location}{folder_for_run}',
                   f'{scenario.log_location}{folder_for_run}/ta_logs']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)


def check_log_folder(scenario, folder_for_run=None):
    """
    Set up log directory.

    Parameters
    ----------
    folder_for_run : str
        Path to log directory.
    """
    if folder_for_run is None:
        folder_for_run = "latest"
    if not os.path.exists(f"{scenario.log_location}"):
        os.makedirs(f"{scenario.log_location}")

    if not os.path.exists(f'{scenario.log_location}{folder_for_run}'):
        os.makedirs(f'{scenario.log_location}{folder_for_run}')

    if not os.path.exists(f'{scenario.log_location}{folder_for_run}/ta_logs'):
        os.makedirs(f'{scenario.log_location}{folder_for_run}/ta_logs')


def save_latest_logs(folder_for_run, scenario):
    """
    Saves latest logs.

    Parameters
    ----------
    folder_for_run : str
        Path to log directory.
    """
    if folder_for_run == "latest":
        shutil.copytree(f'{scenario.log_location}latest', f"{scenario.log_location}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


def log_termination_setting(logger, scenario):
    """
    Log termination criterion for AC process.

    Parameters
    ----------
    logger : logging.logger
        Logging object.
    scenrario : selector.scenario.Scenario
        AC scenario.
    """
    if scenario.termination_criterion == "total_runtime":
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total runtime is: {scenario.wallclock_limit}")
    elif scenario.termination_criterion == "total_tournament_number":
        print(scenario.termination_criterion)
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total number of tournaments is: {scenario.total_tournament_number}")
    else:
        logger.info("No valid termination criterion has been parsed. "
                    "The termination criterion will be set to runtime.")
        logger.info(f"The total runtime is: {scenario.wallclock_limit}")


class TournamentEncoder(json.JSONEncoder):
    """
    Encodes selector.pool.Tournament for logging.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default(self, o):
        """
        Encodes objects in selector.pool.Tournament.

        o : Any
            object from selector.pool.Tournament.
        """
        if dataclasses.is_dataclass(o):
            data_dic = dataclasses.asdict(o)
            if "ray_object_store" in data_dic.keys():
                del data_dic["ray_object_store"]
            return data_dic
        elif isinstance(o, uuid.UUID):
            return str(o)
        elif isinstance(o, Enum):
            return str(o)
        elif isinstance(o, ray._raylet.ObjectRef):
            return str(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            for k in o.keys():
                if isinstance(k, uuid.UUID):
                    o[str(k)] = o.pop(k)
            return o

        return super().default(o)


class ConfEncoder(json.JSONEncoder):
    """
    Encodes selector.pool.Configuration object for logging.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)
