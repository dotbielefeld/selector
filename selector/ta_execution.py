
import logging
import time
import subprocess
import ray
import numpy as np
import psutil
import os

from threading  import Thread
from queue import Queue, Empty


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()

def get_running_processes(ta_process_name):
    processes =[]
    for proc in psutil.process_iter():
        try:
            processName = proc.name()
            processID = proc.pid
            if processName in [ta_process_name]:
                processes.append([processName, processID])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def termination_check(process_pid, process_status, ta_process_name, python_pid, conf_id, instance):
    running_processes = get_running_processes(ta_process_name)

    sr = False
    for rp in running_processes:
        if process_pid == rp[1]:
            sr = True

    if sr:
        logging.info(f"Failed to terminate {conf_id}, {instance}: process {process_pid} with {process_status} on {python_pid} is still running")
    else:
        logging.info(
            f"Successfully terminated {conf_id}, {instance} on {python_pid} with {process_status}")

@ray.remote(num_cpus=1, max_retries=0,  retry_exceptions= False)
def tae_from_cmd_wrapper(conf, instance_path, cache, ta_command_creator, scenario):
    """
    Execute the target algorithm with a given conf/instance pair by calling a user provided Wrapper that created a cmd
    line argument that can be executed
    :param conf: Configuration
    :param instance: Instances
    :param cache: Cache
    :param ta_command_creator: Wrapper that creates a
    :return:
    """
    # todo logging dic should be provided somewhere else -> DOTAC-37
    logging.basicConfig(filename=f'./selector/logs/wrapper_log_for{conf.id}.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    try:
        logging.info(f"Wrapper TAE start {conf}, {instance_path}")
        runargs = {'instance': f'{instance_path}', 'seed': scenario.seed if scenario.seed else -1, "id":f"{conf.id}"}

        # TODO should i also measure the time from the main thread to here?
        cmd = ta_command_creator.get_command_line_args(runargs, conf.conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             close_fds=True)

        # Blocks
        #for line in iter(p.stdout.readline, ''):
        #     line = line.decode("utf-8")
         #    print(line)

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()
        timeout = False
        while p.poll() is None:
            try:
                line = q.get(timeout=.5)
            except Empty:
                pass
            else:
                cache.put_intermediate_output.remote(conf.id, instance_path, line)
                logging.info(f"Wrapper TAE intermediate feedback {conf}, {instance_path} {line}")
            if time.time() - start > scenario.cutoff_time:
                timeout = True
                logging.info(f"Timeout reached, terminating: {conf}, {instance_path} {time.time() - start}")

                p.terminate()
                time.sleep(1)
                if p.poll() is None:
                    p.kill()
                if scenario.ta_pid_name is not None:
                    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)
                    
        ta_end = time.time()
        if timeout:
            cache.put_result.remote(conf.id, instance_path, scenario.cutoff_time)
        else:
            cache.put_result.remote(conf.id, instance_path, ta_end - start)

        time.sleep(0.2)
        logging.info(f"Wrapper TAE end {conf}, {instance_path}")
        return  conf, instance_path, False

    except KeyboardInterrupt:
        logging.info(f" Killing: {conf}, {instance_path} ")
        # We only terminated the subprocess in case it has started (p is defined)
        if 'p' in vars():
            #termination_check(p.pid, p.poll(), "glucose-simp", os.getpid(), conf.id, instance_path)
            p.terminate()
            time.sleep(1)
            if p.poll() is None:
                p.kill()
            if scenario.ta_pid_name is not None:
                termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(), conf.id, instance_path)

        #logging.info(f"Killing status: {p.poll()} {conf.id} {instance_path}")
        #try:
        #    os.killpg(p.pid, signal.SIGTERM)
        #except ProcessLookupError:
        #    pass
        cache.put_result.remote(conf.id, instance_path, np.nan)
        logging.info(f"Killing status: {p.poll()} {conf.id} {instance_path}")
        return  conf, instance_path, True


@ray.remote(num_cpus=1)
def tae_from_aclib(conf, instance, cache, ta_exc):
    pass
# TODO


