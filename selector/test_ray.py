
import logging
import time
import subprocess
import ray
import numpy as np
# import psutil
import os
import signal
import traceback
import re
import copy
import sys
from threading import Thread
from queue import Queue, Empty

sys.stderr = open(os.devnull, "w")
try:
    import psutil
finally:
    sys.stderr = sys.stderr

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()

def get_running_processes(ta_process_name):
    processes = []
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

def tae_from_cmd_wrapper_rt():
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

    try:
        cmd = 'stdbuf -oL /media/dweiss/Transcend5/AC_architecture/Selector/selector/cadical/cadical  --arena=1 --arenacompact=true --arenasort=1 --binary=false --check=false --compact=false --compactint=43502 --compactmin=8601 --elim=false --elimclslim=45931 --elimint=569853 --elimocclim=427 --elimrounds=141 -q /media/dweiss/Transcend5/AC_architecture/Selector/selector/cadical/circuit_fuzz/fuzz_100_28405.cnf'

        p = psutil.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)

        print(p.poll())

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()
        start = time.time()

        timeout = False
        empty_line = False
        memory_p = 0
        cpu_time_p = 0
        reading = True

        while reading:
            try:
                line = q.get(timeout=.5)
                empty_line = False
                print('\ncpu_time_p', cpu_time_p)
                # Get the cpu time and memory of the process
            except Empty:
                empty_line = True
                if p.poll() is None:
                    #cpu_time_p = time.time() - start
                    #print(cpu_time_p)
                    cpu_time_p = p.cpu_times().user
                    print('cpu time 109', cpu_time_p, p.cpu_times())
                    # memory_p = p.memory_info().rss / 1024 ** 2
                    # print(p.memory_info().rss)
                if float(cpu_time_p) > float(10) or float(memory_p) > float(10) and timeout ==False:
                    timeout = True
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)
                pass
            else:  # write intemediate feedback
                print(line)

            if p.poll() is None:
                # Get the cpu time and memory of the process
                try:
                    cpu_time_p = time.time() - start
                    print(cpu_time_p)
                    # cpu_time_p = p.cpu_times().user
                    # memory_p = p.memory_info().rss / 1024 ** 2
                except Exception as e:
                    print("Looking for this", e, p.poll(), cpu_time_p)
                    pass

                if float(cpu_time_p) > float(10) or float(memory_p) > float(
                        10) and timeout == False:
                    timeout = True
                    if p.poll() is None:
                        p.terminate()
                    try:
                        time.sleep(1)
                    except:
                        pass
                    if p.poll() is None:
                        p.kill()
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    # if scenario.ta_pid_name is not None:
                    #    termination_check(p.pid, p.poll(), scenario.ta_pid_name, os.getpid(),conf.id, instance_path)

            # Break the while loop when the ta was killed or finished
            if empty_line and p.poll() != None:
                reading = False

        time.sleep(0.2)

    except Exception:
        print({traceback.format_exc()})
        logging.info(f"Exception in TA execution: {traceback.format_exc()}")

if __name__ == "__main__":

    tae_from_cmd_wrapper_rt()
