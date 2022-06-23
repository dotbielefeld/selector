import os
import shutil
from datetime import datetime

def clear_logs():
    """
    Clear the logs
    """
    for folder in ['./selector/logs/latest' ,'./selector/logs/latest/ta_logs']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

def check_log_folder():
    if not os.path.exists("./selector/logs"):
        os.makedirs("./selector/logs")

    if not os.path.exists('./selector/logs/latest'):
        os.makedirs('./selector/logs/latest')

    if not os.path.exists('./selector/logs/latest/ta_logs'):
        os.makedirs('./selector/logs/latest/ta_logs')

def save_latest_logs():
    shutil.copytree('./selector/logs/latest', f"./selector/logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

def log_termination_setting(logger, scenario):
    if scenario.termination_criterion == "total_runtime":
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total runtime is: {scenario.total_runtime}")
    elif scenario.termination_criterion == "total_tournament_number":
        logger.info(f"The termination criterion is: {scenario.termination_criterion}")
        logger.info(f"The total number of tournaments is: {scenario.total_tournament_number}")
    else:
        logger.info(f"No valid termination criterion has been parsed. "
                    f"The termination criterion will be set to runtime.")
        logger.info(f"The total runtime is: {scenario.total_runtime}")