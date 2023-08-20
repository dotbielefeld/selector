"""This module contains functions for the SMAC surrogate."""

from smac.configspace import ConfigurationSpace
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.scenario.scenario import Scenario
from smac.configspace import Configuration
from ConfigSpace.conditions import InCondition, AndConjunction
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)
from ConfigSpace.forbidden import (
    ForbiddenEqualsClause,
    ForbiddenAndConjunction
)
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import EI, PI
from smac.optimizer.ei_optimization import LocalSearch
from numpy.random import RandomState
from smac.tae import StatusType
import numpy
import uuid
import random
import copy

from selector.pool import ParamType, Generator
from selector.pool import Configuration as SelConfig
from selector.generators.default_point_generator import (
    check_conditionals,
    check_no_goods
)
from selector.generators.random_point_generator import (
    reset_no_goods,
    random_set_conf
)


class SmacSurr():
    """Surrogate from SMAC."""

    def __init__(self, scenario, seed=False):
        """Initialize smac surrogate.

        :param scenario: dict, scenario translated for smac
        :param seed: int, random seed
        """
        if not seed:
            self.seed = False
        else:
            self.seed = seed
        self.param_bounds = {}
        for param in scenario.parameter:
            self.param_bounds[param.name] = param.bound
        self.stats = Stats
        self.rs = RandomState
        self.s = copy.deepcopy(scenario)
        self.scenario, self.config_space, self.types, self.bounds \
            = self.transfom_selector_scenario_for_smac(scenario)
        self.rh = RunHistory()
        self.rh2epm = RunHistory2EPM4Cost(scenario=self.scenario,
                                          num_params=len(self.config_space),
                                          success_states=StatusType)
        self.rafo \
            = RandomForestWithInstances(configspace=self.config_space,
                                        types=self.types,
                                        bounds=self.bounds,
                                        seed=self.seed,
                                        num_trees=10)

        self.aaf = EI(model=self.rafo)
        self.aafpi = PI(model=self.rafo)
        self.afm = LocalSearch(acquisition_function=self.aaf,
                               config_space=self.config_space
                               )

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
        self.neg_cat = {}

        # Setup parameter space.
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

            elif param.type == ParamType.categorical:
                if type(param.bound[0]) is bool:
                    bounds = []
                    for pb in param.bound:
                        if pb is True:
                            bounds.append(True)
                        elif pb is False:
                            bounds.append(False)
                    if param.default is True:
                        default = bounds.index(True)
                    else:
                        default = bounds.index(True)
                else:
                    if param.bound[0].replace('-', '').isdigit():
                        bounds = []
                        # adjust neg categorical parameters for smac surr
                        if '-' in param.bound[0]:
                            add = -1 * int(param.bound[0])
                            bounds = [str(int(i) + add) for i in param.bound]
                            default = str(int(param.default) + add)
                            self.neg_cat[param.name] = add
                        else:
                            for pb in param.bound:
                                bounds.append(str(pb))
                            default = str(param.default)
                    elif '.' in param.bound[0]:
                        if param.bound[0].replace('-', '').replace('.', '')\
                                .isdigit():
                            bounds = []
                            # adjust neg categorical parameters for smac surr
                            if '-' in param.bound[0]:
                                add = -1 * float(param.bound[0])
                                bounds = [str(float(i) + add)
                                          for i in param.bound]
                                default = str(float(param.default) + add)
                                self.neg_cat[param.name] = add
                            else:
                                for pb in param.bound:
                                    bounds.append(str(pb))
                                default = str(param.default)
                    else:
                        bounds = []
                        for pb in param.bound:
                            bounds.append(str(pb))
                        default = str(param.default)

                parameter \
                    = CategoricalHyperparameter(param.name,
                                                bounds,
                                                default_value=default)

                config_space.add_hyperparameter(parameter)

        cond_list = []

        def transform_conditionals(config_space, condvalues):
            if isinstance(config_space[parent],
                          CategoricalHyperparameter):
                if type(condvalues[0]) == str:
                    if condvalues[0].replace('-', '').isdigit():
                        for i, condval in enumerate(condvalues):
                            condvalues[i] = str(condval)
                    elif '.' in condvalues[0]:
                        if condvalues[0].replace('-', '').\
                                replace('.', '').isdigit():
                            for i, condval in enumerate(condvalues):
                                condvalues[i] = str(condval)
                    else:
                        for i, condval in enumerate(condvalues):
                            condvalues[i] = condval

            return condvalues

        # Set up conditionals.
        for child, parents in scenario.conditionals.items():
            if len(parents) >= 2:
                conjunction = []
                for parent, vals in parents.items():
                    condvalues = vals
                    for condval in condvalues:
                        if condval == 'True':
                            condval = True
                        elif condval == 'False':
                            condval = False
                    # adjust neg categorical parameters for smac surr
                    if parent in self.neg_cat:
                        for i, c in enumerate(condvalues):
                            if '.' in c:
                                condvalues[i] = str(float(c) +
                                                    self.neg_cat[parent])
                            else:
                                condvalues[i] = str(int(c) +
                                                    self.neg_cat[parent])

                    condvalues = \
                        transform_conditionals(config_space, condvalues)

                    # Conjunction needed in ConfigSpace if parameter
                    # has more than one conditionals
                    conjunction.append(
                        InCondition(child=config_space[child],
                                    parent=config_space[parent],
                                    values=condvalues))

                cond_list.append(AndConjunction(*conjunction))

            else:
                parent = list(parents.keys())[0]
                vals = list(parents.values())[0]
                condvalues = vals
                for condval in condvalues:
                    if condval == 'True':
                        condval = True
                    elif condval == 'False':
                        condval = False
                # adjust neg categorical parameters for smac surr
                if parent in self.neg_cat:
                    for i, c in enumerate(condvalues):
                        if '.' in c:
                            condvalues[i] = str(float(c) +
                                                self.neg_cat[parent])
                        else:
                            condvalues[i] = str(int(c) + self.neg_cat[parent])

                condvalues = \
                    transform_conditionals(config_space, condvalues)

                cond_list.append(
                    InCondition(child=config_space[child],
                                parent=config_space[parent],
                                values=condvalues))

            config_space.add_conditions(cond_list)

        # Setup no goods.
        for ng in scenario.no_goods:
            ng_list = []
            for param, val in ng.items():

                if isinstance(config_space[param],
                              CategoricalHyperparameter):
                    if type(val) == str:
                        if val.replace('-', '').isdigit():
                            val = str(val)
                        elif '.' in val:
                            if val.replace('-', '').\
                                    replace('.', '').isdigit():
                                val = str(val)
                        else:
                            val = val
                        # adjust neg categorical parameters for smac surr
                        if param in self.neg_cat:
                            if '.' in val:
                                val = str(float(val) + self.neg_cat[param])
                            else:
                                val = str(int(val) + self.neg_cat[param])

                ng_list.append(ForbiddenEqualsClause(config_space[param], val))

            config_space.add_forbidden_clause(
                ForbiddenAndConjunction(*ng_list))

        types, bounds = get_types(config_space)

        # SMAC scenario object
        s = Scenario({'run_obj': 'runtime',
                      'cutoff-time': scenario.cutoff_time,
                      'runcount-limit': 50,
                      'cs': config_space,
                      'deterministic': True,
                      'acq_opt_challengers': 5})

        return s, config_space, types, bounds

    def transform_values(self, conf, pred=False):
        """Transform configuration values in SMAC format.

        :param conf: object, selector.pool.Configuration
        :return config: dict, parameter values
        """
        config = {}

        # Check conditionals and reset parameters if violated
        cond_vio = check_conditionals(self.s, conf.conf)

        for cv in cond_vio:
            conf.conf[cv] = None

        for param in self.config_space:
            if param in conf.conf:
                if conf.conf[param] is None:
                    config[param] = None
                    continue
            else:
                config[param] = None
                continue

            if isinstance(self.config_space[param],
                          UniformFloatHyperparameter):
                config[param] = float(conf.conf[param])

            if isinstance(self.config_space[param],
                          UniformIntegerHyperparameter):
                config[param] = int(conf.conf[param])

            if isinstance(self.config_space[param],
                          CategoricalHyperparameter):
                if isinstance(conf.conf[param], (numpy.bool_, bool)):
                    config[param] = bool(conf.conf[param])
                else:
                    if type(conf.conf[param]) == str or \
                            isinstance(conf.conf[param], numpy.str_):
                        if conf.conf[param].replace('-', '').isdigit():
                            config[param] = str(conf.conf[param])
                        elif '.' in conf.conf[param]:
                            if conf.conf[param].replace('-', '').\
                                    replace('.', '').isdigit():
                                config[param] = str(conf.conf[param])
                        else:
                            if pred:
                                config[param] = \
                                    str(self.param_bounds[param].
                                        index(conf.conf[param]))
                            else:
                                config[param] = conf.conf[param]
                    else:
                        if type(conf.conf[param]) == int or \
                                type(conf.conf[param]) == float:
                            config[param] = conf.conf[param]
                        else:
                            config[param] = str(conf.conf[param])
                    if param in self.neg_cat:
                        if type(config[param]) == float:
                            config[param] = config[param] + self.neg_cat[param]
                        elif type(config[param]) == int:
                            config[param] = config[param] + self.neg_cat[param]
                        elif '.' in config[param]:
                                config[param] = str(float(config[param]) +
                                                    self.neg_cat[param])
                        else:
                            config[param] = str(int(config[param]) +
                                                self.neg_cat[param])

        return config

    def update(self, history, configs, results, terminations):
        """Update SMAC epm.

        :param history: Tournament history
        :param conf: object, selector.pool.Configuration
        :param state: object selector.pool.Status, status of this point
        :param tourn_nr: int, number of tournament, which to update with
        """
        config_dict = {}
        for c in configs:
            config_dict[c.id] = c
        # instances in tournament
        instances = history.instance_set
        for cid in config_dict.keys():
            # config in results
            for ins in instances:
                conf = config_dict[cid]

                if ins in results[cid]:
                    if not numpy.isnan(results[cid][ins]):
                        state = StatusType.SUCCESS

                    elif cid in terminations:
                        state = StatusType.CAPPED

                    else:
                        state = StatusType.TIMEOUT

                    config = self.transform_values(conf)
                    config = dict(sorted(zip(config.keys(),
                                             config.values())))
                    # adjust neg categorical parameters for smac surr
                    for c, v in config.items():
                        if c in self.neg_cat:
                            if type(self.neg_cat[c]) == int:
                                config[c] = str(v)
                            else:
                                config[c] = str(v)

                    config = Configuration(self.config_space, values=config)

                    self.surr.runhistory.add(config, results[cid][ins],
                                             results[cid][ins], state,
                                             self.seed, ins)

    def get_suggestions(self, scenario, n_samples, *args):
        """Get point suggestions from SMAC.

        :param scenario: scenario object from selector
        :param n_samples: int, number of point suggestions to generate
        :return suggestions: list, list of configurations
        """
        suggestions = []
        added = 0
        param_order = []
        params = scenario.parameter
        for p in params:
            param_order.append(p.name)

        # Tell SMAC how many suggetions to make
        self.surr.scenario.acq_opt_challengers = n_samples

        while len(suggestions) < n_samples:
            sugg = self.surr.choose_next()
            for s in sugg:
                if added < n_samples:
                    if not self.seed:
                        identity = uuid.uuid4()
                    else:
                        identity = uuid.UUID(int=random.getrandbits(self.seed))

                    sugg_items = s.get_dictionary()
                    config_setting = {}
                    for po in param_order:
                        if po in sugg_items:
                            config_setting[po] = sugg_items[po]

                    # adjust neg categorical parameters for target algorithms
                    for k, v in config_setting.items():
                        if k in self.neg_cat:
                            if '.' in v:  # for floats
                                config_setting[k] = str(float(v) -
                                                        self.neg_cat[k])
                            else:  # for ints
                                config_setting[k] = str(int(v) -
                                                        self.neg_cat[k])

                    # For consistency, set random value for turned off
                    # parameters (due to conditionals), since SMAC accounts
                    # for conditionals in config generation and all other
                    # generators do not
                    for param in self.s.parameter:
                        if param.name in self.s.conditionals and \
                                param.name not in config_setting:
                            config_setting[param.name] = \
                                random_set_conf([param])[param.name]

                    suggestions.append(
                        SelConfig(identity,
                                  config_setting,
                                  Generator.smac))

                    # Check no goods and reset values if violated
                    ng_vio = check_no_goods(scenario, suggestions[added].conf)
                    while ng_vio:
                        suggestions[added].conf = \
                            reset_no_goods(scenario, suggestions[added].conf)
                        ng_vio = check_no_goods(scenario,
                                                suggestions[added].conf)

                    added += 1

        return suggestions

    def transform_for_epm(self, confs, pred=False):
        """Transform configuration to suit SMAC epm.

        :param confs: list of objects, [selector.pool.Configuration,]
        :param pred: bool, True if used in predict function
        :return configs: list, transformed configurations
        """
        configs = []

        for con in confs:
            config = self.transform_values(con, pred)

            for i, c in config.items():
                if c is None:
                    config[i] = numpy.nan
                # config[i] = float(config[i])

            configs.append(list(config.values()))

        configs = numpy.array(configs, dtype=float)

        return configs

    def predict(self, confs, i):
        """Predict performance/quality of configurations with SMAC epm.

        :param confs: list of objects, [selector.pool.Configuration,]
        :return [mean, var]: numpy.ndarray, mean and variance of predicted
            performance/quality
        """
        all_configs = self.transform_for_epm(confs, pred=True)

        if not any(isinstance(i, list) for i in all_configs):
            all_configs = [all_configs]

        if self.surr.model.rf is not None:

            m = []
            v = []
            for c in all_configs:
                mean, var = self.surr.model._predict(X=c)
                for i, val in enumerate(mean):
                    m.append(mean[i][0])
                    v.append(var[i][0])

            return numpy.array(m), numpy.array(v)

        else:

            return None

    def expected_improvement(self, suggestions, _):
        """Compute expected improvement via SMAC epm.

        :param suggestions: list of objects, [selector.pool.Configuration,]
        :return ei: numpy.ndarray, expected improvements
        """
        configs = self.transform_for_epm(suggestions)

        if self.surr.model.rf is not None:
            ei = self.surr.acquisition_func._compute(X=configs)
            expimp = []

            for e in ei:
                expimp.append(list(e)[0])

            return expimp
        else:
            return [[0] for s in suggestions]

    def probability_improvement(self, suggestions, results, i):
        """Compute probability of improvement.

        :param suggestions: list of objects, [selector.pool.Configuration,]
        :return pi: numpy.ndarray, probabilities of improvement
        """
        best_val = min(min(list(d.values()))
                       for d in list(results.values()))

        self.aafpi.update(eta=best_val, model=self.surr.model)

        if self.aafpi.eta is not None and \
                self.surr.model.rf is not None:
            configs = self.transform_for_epm(suggestions)

            pi = self.aafpi._compute(X=configs)

            probimp = []

            for p in pi:
                probimp.append(list(p))

            return probimp
        else:
            return [[0] for _ in suggestions]
