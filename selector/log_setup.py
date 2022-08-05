import os
import shutil
from datetime import datetime

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
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total number of tournaments is: {scenario.total_tournament_number}")
    else:
        logger.info(f"No valid termination criterion has been parsed. "
                    f"The termination criterion will be set to runtime.")
        logger.info(f"The total runtime is: {scenario.wallclock_limit}")