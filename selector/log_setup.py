import os

def clear_logs():
    """
    Clear the logs
    """
    for folder in ['./selector/logs' ,'./selector/logs/ta_logs']:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

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