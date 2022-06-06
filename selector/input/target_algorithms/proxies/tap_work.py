import time
import logging
import argparse
import math
import random
import numpy
import numpy as np


def target_algorithm_work(worktime, instance, id):
    logging.basicConfig(filename='./selector/logs/ta_logs/tap_work_c{}.logger'.format(worktime), level=logging.INFO,
                        format='%(asctime)s %(message)s')

    print(f"TAE start {id}, {instance}", flush=True)
    logging.info(f"TAE start {id}, {instance}")
    start = time.time()
    worked = 0
    stopers = list(range(1, math.floor(worktime)))
    while worked < worktime:

        sqrt_num = random.getrandbits(128)
        while sqrt_num > 2:
            sqrt_num = math.sqrt(sqrt_num)
            worked = time.time() - start
        if math.floor(worked) in stopers:
            stopers.remove(math.floor(worked))
            print(f"TAE intermediate result {id}, {instance} {math.floor(worked)}", flush=True)
            logging.info(f"TAE intermediate result {id}, {instance} {math.floor(worked)}")


    print(f"TAE end {id}, {instance}", flush=True)
    logging.info(f"TAE end {id}, {instance}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Minimal working example')
    parser.add_argument('-c', '--configurations', required=True, type=float, nargs="+")
    parser.add_argument('-i', '--instance', required=True, type=str, nargs="+")
    parser.add_argument('-ii', '--id', required=True, type=int, nargs="+")
    args = vars(parser.parse_args())

    target_algorithm_work(args["configurations"][0], args["instance"][0], args["id"][0])