import os
import shutil
from datetime import datetime
import dataclasses
import json
import uuid
from enum import Enum
import ray
import numpy as np

def clear_logs(folder_for_run = None):
    """
    Clear the logs
    """
    if folder_for_run == None:
        folder_for_run = "latest"

    for folder in [f'./selector/logs/{folder_for_run}' ,f'./selector/logs/{folder_for_run}/ta_logs']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

def check_log_folder(folder_for_run = None):
    if folder_for_run == None:
        folder_for_run = "latest"
    if not os.path.exists("./selector/logs"):
        os.makedirs("./selector/logs")

    if not os.path.exists(f'./selector/logs/{folder_for_run}'):
        os.makedirs(f'./selector/logs/{folder_for_run}')

    if not os.path.exists(f'./selector/logs/{folder_for_run}/ta_logs'):
        os.makedirs(f'./selector/logs/{folder_for_run}/ta_logs')

def save_latest_logs(folder_for_run):
    if folder_for_run == "latest":
        shutil.copytree('./selector/logs/latest', f"./selector/logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

def log_termination_setting(logger, scenario):
    if scenario.termination_criterion == "total_runtime":
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total runtime is: {scenario.wallclock_limit}")
    elif scenario.termination_criterion == "total_tournament_number":
        print(scenario.termination_criterion)
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total number of tournaments is: {scenario.total_tournament_number}")
    else:
        logger.info(f"No valid termination criterion has been parsed. "
                    f"The termination criterion will be set to runtime.")
        logger.info(f"The total runtime is: {scenario.wallclock_limit}")

class TournamentEncoder(json.JSONEncoder):
    def default(self, o):
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
