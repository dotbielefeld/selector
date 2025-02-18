import ray
import logging
import json
import random
from selector.log_setup import TournamentEncoder


@ray.remote(num_cpus=1)
class TargetAlgorithmObserver:

    def __init__(self, scenario):
        self.intermediate_output = {}
        self.results = {}
        self.start_time = {}
        self.tournament_history = {}
        self.termination_history = {}
        self.tournaments = {}
        self.read_from = {"conf id":1 , "instance_id":1 , "index":1 }
        self.scenario = scenario
        self.core_affinities = {}
        for c in range(2 + self.scenario.tournament_size * self.scenario.number_tournaments):
            self.core_affinities[c] = None

        # todo logging dic should be provided somewhere else -> DOTAC-37
        logging.basicConfig(filename=f'./selector/logs/{self.scenario.log_folder}/Target_Algorithm_Cache.logger', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def get_free_core(self):
        free_cores = [c for c, v in self.core_affinities.items() if v is None]
        if not free_cores:
            return random.choice(list(self.core_affinities.keys()))
        else:
            return random.choice(free_cores)

    def record_core_affinity(self, core, task):
        self.core_affinities[core] = task

    def remove_core_affinity(self, core):
        self.core_affinities[core] = None

    def put_intermediate_output(self, conf_id, instance_id, value):
        logging.info(f"Getting intermediate_output: {conf_id}, {instance_id}, {value} ")

        if conf_id not in self.intermediate_output:
            self.intermediate_output[conf_id] = {}

        if instance_id not in self.intermediate_output[conf_id]:
            self.intermediate_output[conf_id][instance_id] = [value]
        else:
            self.intermediate_output[conf_id][instance_id] = self.intermediate_output[conf_id][instance_id] + [value]

    def get_intermediate_output(self):
        # TODO store from where we have read last and contiue form there
        return self.intermediate_output

    def put_result(self, conf_id, instance_id, result):
        logging.info(f"Getting final result: {conf_id}, {instance_id}, {result}")
        if conf_id not in self.results:
            self.results[conf_id] = {}

        if instance_id not in self.results[conf_id]:
            self.results[conf_id][instance_id] = result

    def get_results(self):
        logging.info(f"Publishing results")
        return self.results

    def get_results_single(self, conf_id, instance_id):
        result = False
        if conf_id in list(self.results.keys()):
            if instance_id in list(self.results[conf_id].keys()):
                result = self.results[conf_id][instance_id]
        return result

    def put_start(self,conf_id, instance_id, start):
        logging.info(f"Getting start: {conf_id}, {instance_id}, {start} ")
        if conf_id not in self.start_time:
            self.start_time[conf_id] = {}

        if instance_id not in self.start_time[conf_id]:
            self.start_time[conf_id][instance_id] = start

    def get_start(self):
        logging.info(f"Publishing start")
        return self.start_time

    def put_tournament_history(self, tournament):
        self.tournament_history[tournament.id ] = tournament

    def get_tournament_history(self):
        return self.tournament_history

    def put_tournament_update(self, tournament):
        self.tournaments[tournament.id] = tournament

    def remove_tournament(self,tournament):
        self.tournaments.pop(tournament.id)

    def get_tournament(self):
        return list(self.tournaments.values())

    def put_termination_history(self, conf_id, instance_id):
        if conf_id not in self.termination_history:
            self.termination_history[conf_id] = []

        if instance_id not in self.termination_history[conf_id]:
            self.termination_history[conf_id].append(instance_id)
        else:
            logging.info("This should not happen: we kill something we already killed")

    def get_termination_history(self):
        return self.termination_history

    def get_termination_single(self, conf_id, instance_id):
        termination = False
        if conf_id in list(self.termination_history.keys()):
            if instance_id in list(self.termination_history[conf_id]):
                termination = True
        return termination

    def save_rt_results(self):
        with open(f"./selector/logs/{self.scenario.log_folder}/run_history.json", 'a') as f:
            history = {str(k):v for k,v in self.results.items()}
            json.dump(history, f, indent=2)

    def save_tournament_history(self):
        with open(f"./selector/logs/{self.scenario.log_folder}/tournament_history.json", 'a') as f:
            history = {str(k): v for k, v in self.tournament_history.items()}
            json.dump(history, f, indent=4, cls=TournamentEncoder)

    def save_tournament_history(self):
        with open(f"./selector/logs/{self.scenario.log_folder}/tournament_history.json", 'a') as f:
            history = {str(k): v for k, v in self.tournament_history.items()}
            json.dump(history, f, indent=4, cls=TournamentEncoder)
