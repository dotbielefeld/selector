import time
import logging
import argparse

def target_algorithm_sleep(sleeptime, instance):
    logging.basicConfig(filename='./selector/logs/ta_logs/tap_sleep_c{}.logger'.format(sleeptime), level=logging.INFO,
                        format='%(asctime)s %(message)s')

    print("start", sleeptime,instance, flush=True)
    logging.info("start {}{}".format(sleeptime, instance))
    time.sleep(sleeptime)

    logging.info("half way {}".format(sleeptime))
    print(sleeptime, "some values", flush=True)

    time.sleep(sleeptime)
    print("end", sleeptime, flush=True)
    logging.info("done {}".format(sleeptime))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Minimal working example')
    parser.add_argument('-c', '--configurations', required=True, type=int, nargs="+")
    parser.add_argument('-i', '--instance', required=True, type=str, nargs="+")
    args = vars(parser.parse_args())
    target_algorithm_sleep(args["configurations"][0], args["instance"][0])
