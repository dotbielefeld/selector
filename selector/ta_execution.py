"""This modules includes functions to execute the target algorithm."""
import logging
import time
import subprocess
import ray
import numpy as np
import psutil
import os
import signal
import traceback
import re
import copy

from threading import Thread
from queue import Queue, Empty
from selector.generators.default_point_generator import check_conditionals

__all__ = ['kill_process_tree', 'cpu_bind_children', 'enqueue_output',
           'get_running_processes', 'tae_from_cmd_wrapper_quality',
           'tae_from_cmd_wrapper_rt', 'termination_check', 'time_measurment']


def kill_process_tree(pid):
    """
    Propagates through the process tree and terminates/kills children.

    Parameters
    ----------
    pid : int
        Process ID of the target algorithm.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()

        _, still_alive = psutil.wait_procs(children, timeout=10)
        for p in still_alive:
            p.kill()

        parent.wait(10)
    except psutil.NoSuchProcess:
        pass


def cpu_bind_children(chosen_core, p, set_affinity, logging):
    """
    Ensures target algorithm and child processes only use the chosen core.

    Parameters
    ----------
    chosen_core : int
        Core the process and children ought to stay on.
    p : object
        Target algorithm process.
    set_affinity : list
        List that tracks CPU affinity of child processes.
    logging : object
        Initialized logging object.
    """
    try:
        if p.poll() is None:
            children = p.children(recursive=True)
            if children:
                for child in children:
                    if (child.is_running() and child.pid 
                            not in set_affinity 
                            and child.cpu_affinity()[0] != chosen_core):
                        try:
                            child.cpu_affinity([chosen_core])
                            set_affinity.append(child.pid)
                            logging.info(
                                f"""New child process {child.pid} 
                                {child.name()} CPU affinity set to core: 
                                {chosen_core}""")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            logging.info(
                                f"""Failed to set CPU affinity for new 
                                child process {child.pid} {child.name()}.""")
    except:
        pass


def time_measurment(p, start, cpu_time_p):
    """
    Measures target algorithm runtime: 1. process tree or 2. process or 3. wall

    Parameters
    ----------
    p : object
        Target algorithm process.
    start : float
        Start time in seconds.
    cpu_time_p : float
        Last runtime measurement.
    """
    cpu_times = p.cpu_times()
    if cpu_times.children_user != 0 and cpu_times.children_user > cpu_time_p:
        cpu_time_p = cpu_times.children_user
    elif cpu_times.user != 0 and cpu_times.user > cpu_time_p:
        cpu_time_p = cpu_times.user
    else:
        cpu_time_p = time.time() - start

    return cpu_time_p


def enqueue_output(out, queue):
    """
    Enqueue output.

    Parameters
    ----------
    out : str
        Target algorithm output.
    queue : multiprocessing.Queue
        Queue to get data.
    """
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()


def get_running_processes(ta_process_name):
    """
    Get list of running processes.

    Parameters
    ----------
    ta_process_name : str
        Name of process to find all processes with.
    """
    processes = []
    for proc in psutil.process_iter():
        try:
            processName = proc.name()
            processID = proc.pid
            if processName in [ta_process_name]:
                processes.append([processName, processID])
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            pass
    return processes


def termination_check(process_pid, process_status, ta_process_name, python_pid,
                      conf_id, instance):
    """
    Check if process was terminated.

    Parameters
    ----------
    process_pid : int
        PID of the process to check.
    process_status : str
        Status of the process.
    ta_process_name : str
        Name of the ta process as noted in system.
    python_pid : int
        PID of the ray actor.
    conf_id : uuid.UUID
        ID of the configuration.
    instance : str
        Problem instance name.
    """
    running_processes = get_running_processes(ta_process_name)

    sr = False
    for rp in running_processes:
        if process_pid == rp[1]:
            sr = True

    if sr:
        logging.info(f"""Failed to terminate {conf_id}, {instance}: process 
            {process_pid} with {process_status} on {python_pid} is still 
            running""")
    else:
        logging.info(
            f"""Successfully terminated {conf_id}, {instance} on {python_pid} 
            with {process_status}""")


@ray.remote(num_cpus=1)
def tae_from_cmd_wrapper_rt(conf, instance_path, cache, ta_command_creator, 
                            scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user-provided Wrapper 
    that creates a command line argument that can be executed.

    Warning
    -------
    If your target algorithms spawn child processes, you might set scenario.cpu_binding = True.

    Parameters
    ----------
    conf : selector.pool.Configuration
        Configuration.
    instance : str
        instance name.
    cache : selector.tournament_dispatcher.MiniTournamentDispatcher
        Cache for all tournament data.
    ta_command_creator : wrapper
        Wrapper that creates a command line.
    scenario : selector.scenario.Scenario
        AC scenario.

    Returns
    -------
    tuple
        - **conf** : object,
          Configuration.
        - **instance_path** : str,
          Path to the instance.
        - **terminated** : bool,
          Whether the process was terminated.
    """

    # todo logging dic should be provided somewhere else -> DOTAC-37
    logging.basicConfig(
        filename=f'{scenario.log_location}{scenario.log_folder}/wrapper_log_for{conf.id}.log',
        level=logging.INFO, format='%(asctime)s %(message)s')

    try:
        logging.info("\n")
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{scenario.instances_dir + instance_path}',
                   'seed': scenario.seed if scenario.seed else -1,
                   "id": f"{conf.id}"}

        clean_conf = copy.copy(conf.conf)
        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)
        cmd = ta_command_creator.get_command_line_args(runargs, clean_conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()

        if scenario.cpu_binding:
            set_affinity = []
            chosen_core = ray.get(cache.get_free_core.remote())
            cache.record_core_affinity.remote(chosen_core,
                                              [conf.id, instance_path])
            logging.info(f'Binding TA to core: {chosen_core}')

            ta_execution_process = psutil.Process()
            ta_execution_process.cpu_affinity([chosen_core])

            p.cpu_affinity([chosen_core])

            cpu_bind_children(chosen_core, p, set_affinity, logging)

        timeout = False
        empty_line = False
        memory_p = 0
        cpu_time_p = 0
        reading = True
        solved = False

        while reading:
            try:
                if scenario.cpu_binding:
                    cpu_bind_children(chosen_core, p, set_affinity, logging)
                line = q.get(timeout=.5)
                empty_line = False
            except Empty:
                empty_line = True

            else:  # write intemediate feedback
                if "placeholder" in line:
                    cache.put_intermediate_output.remote(
                        conf.id, instance_path, line)
                    logging.info(f"""Wrapper TAE intermediate feedback {conf}, 
                        {instance_path} {line}""")

                if scenario.solve_match:
                    if any(sm in line for sm in scenario.solve_match):
                        if scenario.runtime_feedback:
                            time_res = float(
                                re.findall(
                                    f"{scenario.runtime_feedback}", line)[0])
                        solved = True

            if p.poll() is None:
                # Get the cpu time and memory of the process
                cpu_time_p = time_measurment(p, start, cpu_time_p)
                memory_p = p.memory_info().rss / 1024 ** 2

                if (float(cpu_time_p) > float(scenario.cutoff_time) 
                        or float(memory_p) > float(
                        scenario.memory_limit)
                        and timeout is False) or (
                        scenario.solve_match and solved):
                    timeout = float(cpu_time_p) > float(scenario.cutoff_time)
                    logging.info(f"""Timeout or memory reached, terminating: 
                        {conf}, {instance_path} {cpu_time_p}""")
                    kill_process_tree(p.pid)
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() is not None:
                reading = False

        if scenario.runtime_feedback and solved:
            cpu_time_p = time_res

        if timeout:
            cache.put_result.remote(conf.id, instance_path, np.nan)
        elif scenario.solve_match:
            if solved:
                cache.put_result.remote(conf.id, instance_path, cpu_time_p)
            else:
                cache.put_result.remote(conf.id, instance_path, np.nan)
        else:
            cache.put_result.remote(conf.id, instance_path, cpu_time_p)

        if scenario.cpu_binding:
            cache.remove_core_affinity.remote(chosen_core)

        time.sleep(0.2)
        logging.info(
            f"Wrapper TAE end {conf}, {instance_path} at {cpu_time_p}s")
        return conf, instance_path, False

    except KeyboardInterrupt:
        logging.info(f" Killing: {conf}, {instance_path} ")
        # We only terminated the subprocess in case it has started (p is defined)
        if 'p' in vars():
            kill_process_tree(p.pid)
            if p.poll() is None:
                p.terminate()
            try:
                time.sleep(1)
            except:
                print("Got sleep interupt", conf, instance_path)
                pass
            if p.poll() is None:
                p.kill()
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception as e:
                pass
            # if scenario.ta_pid_name is not None:
            #   termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(), conf.id, instance_path)
        if scenario.cpu_binding:
            cache.put_result.remote(conf.id, instance_path, np.nan)
            cache.remove_core_affinity.remote(chosen_core)
        try:
            logging.info(
                f"Killing status: {p.poll()} {conf.id} {instance_path}")
        except:
            pass
        return conf, instance_path, True
    except Exception:
        if scenario.cpu_binding:
            try:
                cache.remove_core_affinity.remote(chosen_core)
            except:
                pass
        print({traceback.format_exc()})
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")


@ray.remote(num_cpus=1)
def tae_from_cmd_wrapper_quality(conf, instance_path, cache,
                                 ta_command_creator, scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user-provided Wrapper 
    that creates a command line argument that can be executed.

    Warning
    -------
    If your target algorithms spawn child processes, you might set scenario.cpu_binding = True.

    Parameters
    ----------
    conf : selector.pool.Configuration
        Configuration.
    instance : str
        instance name.
    cache : selector.tournament_dispatcher.MiniTournamentDispatcher
        Cache for all tournament data.
    ta_command_creator : wrapper
        Wrapper that creates a command line.
    scenario : selector.scenario.Scenario
        AC scenario.

    Returns
    -------
    tuple
        - **conf** : object,
          Configuration.
        - **instance_path** : str,
          Path to the instance.
        - **terminated** : bool,
          Whether the process was terminated.
    """
    logging.basicConfig(filename=f'''{scenario.log_location}{scenario.log_folder}/wrapper_log_for{conf.id}.log''',
                        level=logging.INFO,
                        format='%(asctime)s %(message)s')

    try:
        logging.info("\n")
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{scenario.instances_dir + instance_path}',
                   'seed': scenario.seed if scenario.seed else -1,
                   "id": f"{conf.id}", "timeout": scenario.cutoff_time}

        clean_conf = copy.copy(conf.conf)
        # Check conditionals and turn off parameters if violated
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)

        cmd = ta_command_creator.get_command_line_args(runargs, conf.conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()

        if scenario.cpu_binding:
            set_affinity = []
            chosen_core = ray.get(cache.get_free_core.remote())
            cache.record_core_affinity.remote(
                chosen_core, [conf.id, instance_path])
            logging.info(f'Binding TA to core: {chosen_core}')

            ta_execution_process = psutil.Process()
            ta_execution_process.cpu_affinity([chosen_core])

            p.cpu_affinity([chosen_core])

            cpu_bind_children(chosen_core, p, set_affinity, logging)

        timeout = False
        empty_line = False
        memory_p = 0
        cpu_time_p = 0
        reading = True
        quality = [np.nan]

        while reading:
            try:
                if scenario.cpu_binding:
                    cpu_bind_children(chosen_core, p, set_affinity, logging)
                line = q.get(timeout=.5)
                empty_line = False

                # Get the cpu time and memory of the process
            except Empty:
                empty_line = True
                pass
            else:  # write intemediate feedback
                if "placeholder" in line:
                    cache.put_intermediate_output.remote(
                        conf.id, instance_path, line)
                    logging.info(f"""Wrapper TAE intermediate feedback {conf}, 
                        {instance_path} {line}""")

                if scenario.run_obj == "quality":
                    output_trigger = re.search(scenario.quality_match, line)
                    if output_trigger:
                        quality = re.findall(
                            f"{scenario.quality_extract}", line)

            if p.poll() is None:
                # Get the cpu time and memory of the process
                cpu_time_p = time_measurment(p, start, cpu_time_p)
                memory_p = p.memory_info().rss / 1024 ** 2

                if (float(cpu_time_p) > float(scenario.cutoff_time)
                    or float(memory_p) > float(
                        scenario.memory_limit)
                        and timeout is False) or quality != [np.nan]:
                    timeout = True
                    logging.info(f"""Timeout or memory reached, terminating: 
                        {conf}, {instance_path} {cpu_time_p}""")
                    kill_process_tree(p.pid)
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        print("Got sleep interupt", conf, instance_path)
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() is not None:
                reading = False

        if scenario.cpu_binding:
            cache.put_result.remote(conf.id, instance_path, float(quality[0]))
            cache.remove_core_affinity.remote(chosen_core)

        time.sleep(0.2)
        logging.info(f"Wrapper TAE end {conf}, {instance_path}")
        return conf, instance_path, False
        
    except KeyboardInterrupt:
        logging.info(f" Killing: {conf}, {instance_path} ")
        # We only terminated the subprocess in case it has started (p is defined)
        if 'p' in vars():
            kill_process_tree(p.pid)
            if p.poll() is None:
                p.terminate()
            try:
                time.sleep(1)
            except:
                print("Got sleep interupt", conf, instance_path)
                pass
            if p.poll() is None:
                p.kill()
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception as e:
                pass
            # if scenario.ta_pid_name is not None:
            #   termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(), conf.id, instance_path)
        if scenario.cpu_binding:
            cache.put_result.remote(conf.id, instance_path, np.nan)
            cache.remove_core_affinity.remote(chosen_core)
        try:
            logging.info(
                f"Killing status: {p.poll()} {conf.id} {instance_path}")
        except:
            pass
        return conf, instance_path, True
    except Exception:
        if scenario.cpu_binding:
            try:
                cache.remove_core_affinity.remote(chosen_core)
            except:
                pass
        print({traceback.format_exc()})
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")


@ray.remote(num_cpus=1)
def dummy_task(conf, instance_path, cache):
    time.sleep(2)
    cache.put_result.remote(conf.id, instance_path, np.nan)
    return conf, instance_path, True


@ray.remote(num_cpus=1)
def tae_from_aclib(conf, instance, cache, ta_exc):
    pass
# TODO
