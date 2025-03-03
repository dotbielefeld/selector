"""This module contains selectors cache object."""
import ray
import logging
import json
import random
from selector.log_setup import TournamentEncoder


@ray.remote(num_cpus=1)
class TargetAlgorithmObserver:
    """
    Stores and manages all data from tournaments.

    Parameters
    ----------
    scenario : selector.scenario.Scenario
        AC scenario.
    """
    def __init__(self, scenario):
        self.intermediate_output = {}
        self.results = {}
        self.start_time = {}
        self.tournament_history = {}
        self.termination_history = {}
        self.tournaments = {}
        self.read_from = {"conf id": 1, "instance_id": 1, "index": 1}
        self.scenario = scenario
        self.core_affinities = {}
        for c in range(2 + self.scenario.tournament_size * self.scenario.number_tournaments):
            self.core_affinities[c] = None

        # todo logging dic should be provided somewhere else -> DOTAC-37
        logging.basicConfig(filename=f'{self.scenario.log_location}{self.scenario.log_folder}/Target_Algorithm_Cache.logger', level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def get_free_core(self):
        """
        Looks for and returns the index of a free core.

        Returns
        -------
        int
            Index of a free core.
        """
        free_cores = [c for c, v in self.core_affinities.items() if v is None]
        if not free_cores:
            return random.choice(list(self.core_affinities.keys()))
        else:
            return random.choice(free_cores)

    def record_core_affinity(self, core, task):
        """
        Records that a core is bound.

        Parameters
        ----------
        core : int
            Index of core.
        task: dict
            Configuration/instance pair.
        """
        self.core_affinities[core] = task

    def remove_core_affinity(self, core):
        """
        Records that a binding is removed.

        Parameters
        ----------
        core : int
            Index of core.
        """
        self.core_affinities[core] = None

    def put_intermediate_output(self, conf_id, instance_id, value):
        """
        Saves intermediate output.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration.
        instance_id : str
            Name of the instance.
        value : str
            intermediate output of the target algorithm.
        """
        logging.info(f"Getting intermediate_output: {conf_id}, {instance_id}, {value} ")

        if conf_id not in self.intermediate_output:
            self.intermediate_output[conf_id] = {}

        if instance_id not in self.intermediate_output[conf_id]:
            self.intermediate_output[conf_id][instance_id] = [value]
        else:
            self.intermediate_output[conf_id][instance_id] = self.intermediate_output[conf_id][instance_id] + [value]

    def get_intermediate_output(self):
        """Reads intermediate output from target algorithm."""
        # TODO store from where we have read last and contiue form there
        return self.intermediate_output

    def put_result(self, conf_id, instance_id, result):
        """
        Saves result of the configuration/instance run.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration.
        instance_id : str
            Name of the instance.
        result : float
            Result of the configuration/instance run.
        """
        logging.info(f"Getting final result: {conf_id}, {instance_id}, {result}")
        if conf_id not in self.results:
            self.results[conf_id] = {}

        if instance_id not in self.results[conf_id]:
            self.results[conf_id][instance_id] = result

    def get_results(self):
        """Get all results of tournament.

        Returns
        -------
        dict
            Configuration /instance pair results.
        """
        logging.info("Publishing results")
        return self.results

    def get_results_single(self, conf_id, instance_id):
        """Get a single result of tournament.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration.
        instance_id : str
            Name of the problem instance.

        Returns
        -------
        dict
            Configuration /instance pair results.
        """
        result = False
        if conf_id in list(self.results.keys()):
            if instance_id in list(self.results[conf_id].keys()):
                result = self.results[conf_id][instance_id]
        return result

    def put_start(self, conf_id, instance_id, start):
        """Record start of target algorithm run.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration.
        instance_id : str
            Name of the problem instance.
        start : int
            Start time of the run.
        """
        logging.info(f"Getting start: {conf_id}, {instance_id}, {start} ")
        if conf_id not in self.start_time:
            self.start_time[conf_id] = {}

        if instance_id not in self.start_time[conf_id]:
            self.start_time[conf_id][instance_id] = start

    def get_start(self):
        """Returns start time.

        Returns
        -------
        int
            Start time.
        """
        logging.info("Publishing start")
        return self.start_time

    def put_tournament_history(self, tournament):
        """
        Saves tournament to the tournament history.

        Parameters
        ----------
        tournament : selector.pool.Tournament
            Tournament data to save to history.
        """
        self.tournament_history[tournament.id] = tournament

    def get_tournament_history(self):
        """
        Returns tournament history.

        Returns
        ----------
        dict 
            Tournament history: {selector.pool.Tournament.id: selector.pool.Tournament}
        """
        return self.tournament_history

    def put_tournament_update(self, tournament):
        """
        Saves currently running tournament to the tournament history.

        Parameters
        ----------
        tournament : selector.pool.Tournament
            Tournament data to save to history.
        """
        self.tournaments[tournament.id] = tournament

    def remove_tournament(self, tournament):
        """
        Deletes tournament from the tournament history.

        Parameters
        ----------
        tournament : selector.pool.Tournament
            Tournament data to delete.
        """
        self.tournaments.pop(tournament.id)

    def get_tournament(self):
        """
        Get all tournaments.

        Returns
        -------
        list
            List of selector.pool.Tournament records.
        """
        return list(self.tournaments.values())

    def put_termination_history(self, conf_id, instance_id):
        """Save termination history.

        Parameters
        ----------
        conf_id : uuid.UUID
            ID of the configuration.
        instance_id : str
            Name of the problem instance.
        """
        if conf_id not in self.termination_history:
            self.termination_history[conf_id] = []

        if instance_id not in self.termination_history[conf_id]:
            self.termination_history[conf_id].append(instance_id)
        else:
            logging.info("This should not happen: we kill something we already killed")

    def get_termination_history(self):
        """
        Returns termination history.

        Returns
        -------
        dict
            Configuration and instance pairs that were terminated.
        """
        return self.termination_history

    def get_termination_single(self, conf_id, instance_id):
        """
        Returns termination history for a configuration.

        Returns
        -------
        list
            Problem instance names the configuration was terminated on.
        """
        termination = False
        if conf_id in list(self.termination_history.keys()):
            if instance_id in list(self.termination_history[conf_id]):
                termination = True
        return termination

    def save_rt_results(self):
        """Saves results to file."""
        with open(f"./selector/logs/{self.scenario.log_folder}/run_history.json", 'a') as f:
            history = {str(k): v for k, v in self.results.items()}
            json.dump(history, f, indent=2)

    def save_tournament_history(self):
        """Saves tournament history to file."""
        with open(f"./selector/logs/{self.scenario.log_folder}/tournament_history.json", 'a') as f:
            history = {str(k): v for k, v in self.tournament_history.items()}
            json.dump(history, f, indent=4, cls=TournamentEncoder)
