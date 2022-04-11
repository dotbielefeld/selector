
import logging
import time
import subprocess
import ray
import numpy as np


from threading  import Thread
from queue import Queue, Empty


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        line = line.decode("utf-8")
        queue.put(line)
    out.close()

@ray.remote(num_cpus=1)
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
        logging.info(f"Starting ta execution {conf}, {instance_path}")
        runargs = {'instance': f'{instance_path}', 'seed': scenario.seed if scenario.seed else -1}

        cmd = ta_command_creator.get_command_line_args(runargs, conf.conf)
        start = time.time()
        cache.put_start.remote(conf.id, instance_path, start)

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             close_fds=True)

        # Blocks
        # for line in iter(p.stdout.readline, ''):
        #     line = line.decode("utf-8")

        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True
        t.start()
        while p.poll() is None:
            try:
                line = q.get(timeout=.5)
            except Empty:
                pass
            else:
                cache.put_intermediate_output.remote(conf.id, instance_path, line)
                logging.info(f"ta feedback: {time.asctime(time.localtime())}, {line}")

        cache.put_result.remote(conf.id, instance_path, time.time() - start)
        logging.info(f"ta execution finished {conf}, {instance_path}")
        return  conf, instance_path, False

    except KeyboardInterrupt:
        logging.info(f" Killing: {conf.id} {instance_path} ")
        p.terminate()
        time.sleep(1)
        p.kill()
        #try:
        #    os.killpg(p.pid, signal.SIGTERM)
        #except ProcessLookupError:
        #    pass
        logging.info(f"Killing status: {p.poll()} {conf.id} {instance_path}")

        return  conf, instance_path, True


@ray.remote(num_cpus=1)
def tae_from_aclib(conf, instance, cache, ta_exc):
    pass
# TODO


