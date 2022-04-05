import ray
import logging


@ray.remote(num_cpus=1)
class TargetAlgorithmObserver:

    def __init__(self):
        self.intermediate_output = {}
        self.results = {}
        self.start_time = {}
        self.tournament_history = {}
        self.read_from = {"conf id":1 , "instance_id":1 , "index":1 }

        # todo logging dic should be provided somewhere else -> DOTAC-37
        logging.basicConfig(filename='./selector/logs/Target_Algorithm_Cache.log', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def put_intermediate_output(self, conf_id, instance_id, value):

        if conf_id not in self.intermediate_output:
            self.intermediate_output[conf_id] = {}

        if instance_id not in self.intermediate_output[conf_id]:
            self.intermediate_output[conf_id][instance_id] = [value]
        else:
            self.intermediate_output[conf_id][instance_id] = self.intermediate_output[conf_id][instance_id] + [value]


    def get_intermediate_output(self):
        # TODO store from where we have read last and contiue form there
        return self.intermediate_output

    def put_result(self,conf_id, instance_id, result):
        if conf_id not in self.results:
            self.results[conf_id] = {}

        if instance_id not in self.results[conf_id]:
            self.results[conf_id][instance_id] = result

        logging.info(f"Putting results: {conf_id}, {instance_id}, {result} ")
        logging.info(f"current results: {self.results}")

    def get_results(self):
        return self.results

    def put_start(self,conf_id, instance_id, start):

        if conf_id not in self.start_time:
            self.start_time[conf_id] = {}

        if instance_id not in self.start_time[conf_id]:
            self.start_time[conf_id][instance_id] = start

    def get_start(self):
        logging.info(f"Monitor getting start")
        return self.start_time

    def put_tournament_history(self, tournament):
        self.tournament_history[tournament.id ] = tournament

    def get_tournament_history(self):
        return self.tournament_history