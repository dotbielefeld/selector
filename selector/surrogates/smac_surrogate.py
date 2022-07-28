"""This module contains functions for the SMAC surrogate."""

from smac.configspace import ConfigurationSpace
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.scenario.scenario import Scenario
from smac.configspace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.acquisition import EI, PI
from smac.optimizer.ei_optimization import LocalSearch
from numpy.random import RandomState
from smac.tae import StatusType
import numpy
import uuid
import random

from selector.pool import ParamType, Generator, Status
from selector.pool import Configuration as SelConfig


class SmacSurr():
    """Surrogate from SMAC."""

    def __init__(self, scenario, seed=False):
        """Initialize smac surrogate.

        :param scenario: dict, scenario translated for smac
        :param seed: int, random seed
        """
        if not seed:
            self.seed = numpy.random.randint(2**32 - 1)
        else:
            self.seed = seed
        self.stats = Stats
        self.rs = RandomState
        self.scenario, self.config_space, self.types, self.bounds \
            = self.transfom_selector_scenario_for_smac(scenario)
        self.rh = RunHistory()
        self.rh2epm = RunHistory2EPM4Cost(scenario=self.scenario,
                                          num_params=len(self.config_space),
                                          success_states=StatusType)
        self.rafo = RandomForestWithInstances(configspace=self.config_space,
                                              types=self.types,
                                              bounds=self.bounds,
                                              seed=self.seed)
        self.aaf = EI(model=self.rafo)
        self.aafpi = PI(model=self.rafo)
        self.afm = LocalSearch(acquisition_function=self.aaf,
                               config_space=self.config_space)

        self.surr = EPMChooser(scenario=self.scenario,
                               stats=self.stats,
                               runhistory=self.rh,
                               runhistory2epm=self.rh2epm,
                               model=self.rafo,
                               acq_optimizer=self.afm,
                               acquisition_func=self.aaf,
                               rng=self.rs)

    def transfom_selector_scenario_for_smac(self, scenario):
        """Transform scenario to SMAC frmulation.

        :param scenario: scenario object from selector
        :return scenario: scenario object from SMAC
        """
        config_space = ConfigurationSpace()
        types = []
        bounds = []

        for param in scenario.parameter:
            if param.scale:
                log = True
            else:
                log = False

            if param.type == ParamType.integer:
                if param.bound[0] < 0:
                    log = False
                parameter \
                    = UniformIntegerHyperparameter(param.name,
                                                   param.bound[0],
                                                   param.bound[1],
                                                   default_value=param.default,
                                                   log=log)

                config_space.add_hyperparameter(parameter)
                types.append(0)
                bounds.append((param.bound[0], param.bound[1]))

            elif param.type == ParamType.continuous:
                if param.bound[0] < 0:
                    log = False
                parameter \
                    = UniformFloatHyperparameter(param.name,
                                                 param.bound[0],
                                                 param.bound[1],
                                                 default_value=param.default,
                                                 log=log)

                config_space.add_hyperparameter(parameter)
                types.append(0)
                bounds.append((param.bound[0], param.bound[1]))

            elif param.type == ParamType.categorical:
                parameter \
                    = CategoricalHyperparameter(param.name,
                                                param.bound,
                                                default_value=param.default)

                config_space.add_hyperparameter(parameter)
                types.append(len(param.bound))
                bounds.append((len(param.bound), numpy.nan))

        s = Scenario({'run_obj': 'runtime',
                      'cutoff-time': scenario.cutoff_time,
                      'runcount-limit': 50,
                      'cs': config_space,
                      'deterministic': True})

        return s, config_space, types, bounds

    def transform_values(self, conf):
        """Transform configuration values in SMAC format.

        :param conf: object, selector.pool.Configuration
        :return config: dict, parameter values
        """
        config = {}

        for param in self.config_space:
            if param not in conf.conf:
                conf.conf[param] \
                    = self.config_space[param].default_value

            if isinstance(self.config_space[param],
                          UniformFloatHyperparameter):
                config[param] = float(conf.conf[param])

            if isinstance(self.config_space[param],
                          UniformIntegerHyperparameter):
                config[param] = int(conf.conf[param])

            if isinstance(self.config_space[param],
                          CategoricalHyperparameter):
                if isinstance(conf.conf[param], (numpy.bool_, bool)):
                    config[param] \
                        = int(list(
                            self.config_space[param].choices)
                        .index(conf.conf[param]))
                else:
                    config[param] = str(conf.conf[param])

        return config

    def update(self, history, conf, state, tourn_nr):
        """Update SMAC epm.

        :param history: Tournament history
        :param conf: object, selector.pool.Configuration
        :param state: object selector.pool.Status, status of this point
        :param tourn_nr: int, number of tournament, which to update with
        """
        if tourn_nr in history[conf.id]:
            if state == Status.win:
                status = StatusType.SUCCESS

            elif state == Status.cap:
                status = StatusType.CAPPED

            elif state == Status.timeout:
                status = StatusType.TIMEOUT

            elif state == Status.stop:
                status = StatusType.STOP

            elif state == Status.running:
                status = StatusType.RUNNING

            else:
                status = StatusType.RUNNING

            config = self.transform_values(conf)

            config = Configuration(self.config_space, values=config)

            self.surr.runhistory.add(config,
                                     history[conf.id][tourn_nr],
                                     history[conf.id][tourn_nr],
                                     status)

    def get_suggestions(self, scenario, n_samples=8):
        """Get point suggestions from SMAC.

        :param scenario: scenario object from selector
        :param n_samples: int, number of point suggestions to generate
        :return suggestions: list, list of configurations
        """
        suggestions = []
        config_setting = {}
        added = 0
        param_order = []

        params = scenario.parameter
        for p in params:
            param_order.append(p.name)

        while len(suggestions) < n_samples:
            sugg = self.surr.choose_next()
            for s in sugg:
                if added < n_samples:
                    if not self.seed:
                        identity = uuid.uuid4()
                    else:
                        identity = uuid.UUID(int=random.getrandbits(self.seed))

                    sugg_items = s.get_dictionary()
                    for po in param_order:
                        if po in sugg_items:
                            config_setting[po] = sugg_items[po]

                    suggestions.append(
                        SelConfig(identity,
                                  config_setting,
                                  Generator.smac))
                    added += 1

        return suggestions

    def transform_for_epm(self, confs):
        """Transform configuration to suit SMAC epm.

        :param confs: list of objects, [selector.pool.Configuration,]
        :return configs: list, transformed configurations
        """
        configs = []

        for con in confs:

            config = self.transform_values(con)

            for c in config.keys():
                if isinstance(self.config_space[c],
                              CategoricalHyperparameter):
                    if isinstance(config[c], str):
                        config[c] \
                            = int(list(
                                self.config_space[c].choices)
                            .index(config[c]))
            configs.append(numpy.array(list(config.values())))

        configs = numpy.array(configs)

        return configs

    def predict(self, confs):
        """Predict performance/quality of configurations with SMAC epm.

        :param confs: list of objects, [selector.pool.Configuration,]
        :return [mean, var]: numpy.ndarray, mean and variance of predicted
            performance/quality
        """
        configs = self.transform_for_epm(confs)

        return self.surr.model._predict(X=configs)

    def expected_improvement(self, suggestions):
        """Compute expected improvement via SMAC epm.

        :param suggestions: list of objects, [selector.pool.Configuration,]
        :return ei: numpy.ndarray, expected improvements
        """
        configs = self.transform_for_epm(suggestions)

        return self.surr.acquisition_func._compute(X=configs)

    def probability_improvement(self, suggestions):
        """Compute probability of improvement.

        :param suggestions: list of objects, [selector.pool.Configuration,]
        :return pi: numpy.ndarray, probabilities of improvement
        """
        configs = self.transform_for_epm(suggestions)

        return self.aafpi._compute(X=configs)
