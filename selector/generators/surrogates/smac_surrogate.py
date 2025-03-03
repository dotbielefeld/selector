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
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
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
import math
import sys

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from smac.configspace import convert_configurations_to_array
from typing import List, Optional, Tuple
import itertools

from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()


__all__ = ['SmacSurr']


# Adjust SMAC's LocalSearch to only suggest as many points as we actually ask
class LocalSearch(LocalSearch):
    """Implementation of SMAC's local search."""

    def _get_initial_points(
        self,
        num_points: int,
        runhistory: RunHistory,
        additional_start_points: Optional[List[Tuple[float, Configuration]]],
    ) -> List[Configuration]:

        if runhistory.empty():
            init_points = \
                self.config_space.sample_configuration(size=num_points)
        else:
            # initiate local search
            configs_previous_runs = runhistory.get_all_configs()

            # configurations with the highest previous EI
            configs_previous_runs_sorted = \
                self._sort_configs_by_acq_value(configs_previous_runs)
            configs_previous_runs_sorted = \
                [conf[1] for conf in configs_previous_runs_sorted[:num_points]]

            # configurations with the lowest predictive cost,
            # check for None to make unit tests work
            if self.acquisition_function.model is not None:
                conf_array = \
                    convert_configurations_to_array(configs_previous_runs)
                costs = self.acquisition_function.model.\
                    predict_marginalized_over_instances(conf_array)[0]
                assert len(conf_array) == len(costs), (conf_array.shape,
                                                       costs.shape)

                # In case of the predictive model returning the prediction
                # for more than one objective per configuration
                # (for example multi-objective or EIPS) it is not immediately
                # clear how to sort according to the cost
                # of a configuration. Therefore, we simply follow the ParEGO#
                # approach and use a random scalarization.
                if len(costs.shape) == 2 and costs.shape[1] > 1:
                    weights = numpy.array([self.rng.rand()
                                          for _ in range(costs.shape[1])])
                    weights = weights / numpy.sum(weights)
                    costs = costs @ weights

                # From here
                # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
                random = self.rng.rand(len(costs))
                # Last column is primary sort key!
                indices = numpy.lexsort((random.flatten(), costs.flatten()))

                # Cannot use zip here because the indices array cannot index
                # the rand_configs list, because the second is a
                # pure python list
                configs_previous_runs_sorted_by_cost = \
                    [configs_previous_runs[ind] 
                     for ind in indices][:num_points]
            else:
                configs_previous_runs_sorted_by_cost = []

            if additional_start_points is not None:
                additional_start_points = \
                    [asp[1] for asp in additional_start_points[:num_points]]
            else:
                additional_start_points = []

            init_points = []
            init_points_as_set = set()  # type: Set[Configuration]
            for cand in itertools.chain(
                configs_previous_runs_sorted,
                configs_previous_runs_sorted_by_cost,
                additional_start_points,
            ):
                if cand not in init_points_as_set:
                    init_points.append(cand)
                    init_points_as_set.add(cand)

            # Here actual adjustment to give only num_points initial points.
            # We remove math.floor(remove num_pomits / 2)
            # configs_previous_runs_sorted, getting 50/50 from
            # configs_previous_runs_sorted and 
            # configs_previous_runs_sorted_by_cost but favor the former,
            # if odd number.
            if len(init_points) > num_points:
                remove_nr = math.floor((len(init_points) - num_points))
                if remove_nr > math.floor(num_points / 2):
                    remove_nr = math.floor(num_points / 2)
                for rr in range(remove_nr):
                    del init_points[num_points - 1 - rr]
                if len(init_points) > num_points:
                    init_points = init_points[:num_points]

        return init_points


class SmacSurr():
    """Surrogate from SMAC.

    Note
    ----
    Implementation is using source code of the package smac.

    Parameters
    ----------
    scenario : selector.scenario.Scenario
        AC scenario.
    seed : int
        Random seed.
    pca_dim : int
        PCA dimension for SMAC surrogates instance feature PCA.
    """

    def __init__(self, scenario, seed=False, pca_dim=8):
        if not seed:
            self.seed = False
        else:
            self.seed = seed

        # Control How many threads/cores numpy and scipy use
        @controller.wrap(limits=scenario.tournament_size,
                         user_api='openmp')
        @controller.wrap(limits=scenario.tournament_size,
                         user_api='blas')
        def threaded_init(scenario, pca_dim):
            self.pca_dim = pca_dim
            self.best_val = sys.maxsize
            self.param_bounds = {}
            for param in scenario.parameter:
                self.param_bounds[param.name] = param.bound
            self.stats = Stats
            self.rs = RandomState
            self.s = copy.deepcopy(scenario)
            inst_feats = numpy.array(list(scenario.features.values()))
            self.inst_ids = list(scenario.features.keys())
            self.selector_scenario = scenario
            scaler = MinMaxScaler()
            inst_feats = scaler.fit_transform(inst_feats)
            inst_feats = numpy.nan_to_num(inst_feats)
            pca = PCA(n_components=self.pca_dim)
            pca_inst_feats = pca.fit_transform(inst_feats)
            self.pca_inst_feats, self.inst_ids = \
                self.pca_inst_feat_file(self.selector_scenario.instance_file,
                                        self.selector_scenario.feature_file,
                                        self.inst_ids, pca_inst_feats)
            self.scenario, self.config_space, self.types, self.bounds \
                = self.transfom_selector_scenario_for_smac(self.selector_scenario)
            self.rh = RunHistory(overwrite_existing_runs=True)
            self.rh2epm = RunHistory2EPM4LogCost(scenario=self.scenario,
                                                 num_params=len(self.config_space),
                                                 success_states=StatusType)
            self.rafo \
                = RandomForestWithInstances(configspace=self.config_space,
                                            types=self.types,
                                            bounds=self.bounds,
                                            seed=self.seed,
                                            instance_features=self.pca_inst_feats,
                                            num_trees=10,  # Same as in GGApp
                                            pca_components=self.pca_dim)
            self.aaf = EI(model=self.rafo)
            self.aafpi = PI(model=self.rafo)
            self.afm = LocalSearch(acquisition_function=self.aaf,
                                   config_space=self.config_space,
                                   max_steps=10,
                                   n_steps_plateau_walk=1,
                                   vectorization_min_obtain=2,
                                   vectorization_max_obtain=64
                                   )
            self.surr = EPMChooser(scenario=self.scenario,
                                   stats=self.stats,
                                   runhistory=self.rh,
                                   runhistory2epm=self.rh2epm,
                                   model=self.rafo,
                                   acq_optimizer=self.afm,
                                   acquisition_func=self.aaf,
                                   rng=self.rs,
                                   random_configuration_chooser=None,
                                   predict_x_best=False)

        threaded_init(scenario, pca_dim)

    def transfom_selector_scenario_for_smac(self, scenario):
        """Transform scenario to SMAC formulation.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        Returns
        -------
        tuple
            - **s** : smac.scenario,
              AC scenario in SMAC format.
            - **config_space** : smac.configspace.ConfigurationSpace,
              Parameter space definition in SMAC format.
            - **types** : list,
              Parameter types.
            - **bounds** : list of float,
              Parameter bounds.
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

        self.cutoff_time = scenario.cutoff_time

        # SMAC scenario object
        s = Scenario({'run_obj': scenario.run_obj,
                      'cutoff': scenario.cutoff_time,
                      'runcount-limit': 10,
                      'cs': config_space,
                      'deterministic': True,
                      'acq_opt_challengers': scenario.tournament_size,
                      'instance_file': scenario.instance_file,
                      'feature-file': scenario.feature_file})

        return s, config_space, types, bounds

    def transform_values(self, conf, pred=False):
        """Transform configuration values in SMAC format.

        Parameters
        ----------
        conf : selector.pool.Configuration
            Configuration to be transformed.
        pred : bool
            True if configuration is prepared for prediction.
        Returns
        -------
        dict
            Transformed configuration values.
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

    def update(self, history, configs, results, terminations, ac_runtime=None):
        """Update SMAC epm.

        Parameters
        ----------
        history : list of selector.pool.Tournament
            Tournament history.
        configs : list of selector.pool.Configuration
            Configurations that participated in the tournament.
        results : dict
            Results of the tournament.
        terminations : dict
            Information about terminations of runs that occurred.
        ac_runtime : int
            Total AC runtime in seconds so far.
        """
        config_dict = {}
        for c in configs:
            config_dict[c.id] = c
        # instances in tournament
        instances = history.instance_set

        if ac_runtime >= self.selector_scenario.wallclock_limit * 0.15 and \
                (len(instances) * len(config_dict)) * 2 \
                < len(self.surr.runhistory.data):
            ds = (len(instances) * len(config_dict))
            to_delete = []
            delete_by_ids = []
            for rd in self.surr.runhistory.data.keys():
                if (rd not in
                        to_delete and len(delete_by_ids) < ds) or \
                        (rd not in to_delete and rd.config_id in 
                            delete_by_ids):
                    to_delete.append(rd)
                    if rd.config_id not in \
                            delete_by_ids and \
                            len(delete_by_ids) < ds:
                        delete_by_ids.append(rd.config_id)

            for td in to_delete:
                del self.surr.runhistory.data[td]

            for td in delete_by_ids:
                dbc = self.surr.runhistory.ids_config[td]
                del self.surr.runhistory.ids_config[td]
                del self.surr.runhistory.config_ids[dbc]

        for cid in config_dict.keys():
            # config in results
            for ins in instances:
                # OMIT every censored date in update
                if cid in terminations:
                    if ins in terminations[cid]:
                        continue
                conf = config_dict[cid]

                if ins in results[cid]:
                    if not numpy.isnan(results[cid][ins]):
                        state = StatusType.SUCCESS

                    else:
                        # This conf/inst pair was a time limit reach
                        state = StatusType.TIMEOUT
                        results[cid][ins] = self.cutoff_time

                    config = self.transform_values(conf)
                    config = dict(sorted(zip(config.keys(),
                                             config.values())))
                    # adjust neg categorical parameters for smac surr
                    for c, v in config.items():
                        if c in self.neg_cat:
                            if type(self.neg_cat[c]) is int:
                                config[c] = str(v)
                            else:
                                config[c] = str(v)

                    config = Configuration(self.config_space, values=config)

                    self.surr.runhistory.add(config, results[cid][ins],
                                             results[cid][ins], state,
                                             ins, self.seed)

    def get_suggestions(self, scenario, n_samples, *args):
        """
        Suggest configurations to run next based on the next instance set to run on.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        n_samples : int
            Number of configurations to return.
        *args : Any
            Catches unneeded arguments due to the implementation of other surrogates.
        Returns
        -------
        list of selector.pool.Configuration
            Suggested configurations.
        """
        suggestions = []
        added = 0
        param_order = []
        params = scenario.parameter
        for p in params:
            param_order.append(p.name)

        import time

        start = time.time()

        # Tell SMAC how many suggetions to make
        self.surr.scenario.acq_opt_challengers = n_samples

        # Tell SMAC features of the next instances
        if args[2] is not None:
            next_inst_feats = []
            for inst in args[2]:
                next_inst_feats.append(self.pca_inst_feats[self.inst_ids.index(inst)])
            self.surr.model.instance_features = numpy.array(next_inst_feats)
        
        while len(suggestions) < n_samples:
            sugg = self.surr.choose_next()
            sugg = list(sugg)

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

        Parameters
        ----------
        confs : list selector.pool.Configuration
            Configurations to be transformed.
        pred : bool
            True if prepared for prediction.
        Returns
        -------
        ndarray
            Transformed configurations.
        """
        configs = []

        for con in confs:
            config = self.transform_values(con, pred)

            for i, c in config.items():
                if c is None:
                    config[i] = numpy.nan

            configs.append(list(config.values()))

        configs = numpy.array(configs, dtype=float)

        return configs

    def predict(self, confs, inst):
        """
        Predict performance/quality of configurations with GGA++ EPM.

        Parameters
        ----------
        confs : list of selector.pool.Configuration
            Suggested configurations.
        inst : list 
            List of next instances to run the tournament on.
        Returns
        -------
        tuple
            - ndarray,
              Mean of predicted performance/quality.
            - ndarray,
              Variance of predicted performance/quality.
        """
        all_configs = self.transform_for_epm(confs, pred=True)

        if not any(isinstance(i, list) for i in all_configs):
            all_configs = [all_configs]

        if self.surr.model.rf is not None:

            m = []
            v = []

            m_av = 0
            v_av = 0

            n_samples, _, _ = self.surr._collect_data_to_train_model()
            
            for c in all_configs[0]:
                if n_samples.shape[0] > self.pca_dim:
                    for infe in inst:
                        instfeat = numpy.asarray(
                            self.pca_inst_feats[self.inst_ids.index(infe)])
                        mean, var = \
                            self.surr.model._predict(
                                X=numpy.array([numpy.append(c, instfeat)]))
                        for i, val in enumerate(mean):
                            m_av += mean[i][0]
                            v_av += var[i][0]

                        m.append(m_av / len(inst))
                        v.append(v_av / len(inst))
                else:
                    mean, var = \
                        self.surr.model._predict(X=numpy.array([c]))
                    for i, val in enumerate(mean):
                        m += mean[i][0]
                        v += var[i][0]

            return numpy.array(m), numpy.array(v)

        else:

            return None

    def expected_improvement(self, suggestions, _):
        """
        Compute expected improvement via SMAC model.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        _ : list of str
            List of next instances to be run.
        Returns
        -------
        ndarray
            **ei**: Expected improvements.
        """
        configs = self.transform_for_epm(suggestions)

        if self.surr.model.rf is not None:
            self.surr.acquisition_func.update(eta=self.best_val,
                                              model=self.surr.model)
            ei = self.surr.acquisition_func._compute(X=configs)
            expimp = []

            for e in ei:
                expimp.append(list(e)[0])

            return expimp
        else:
            return [[0] for s in suggestions]

    def probability_improvement(self, suggestions, results, i):
        """
        Compute probability of improvement.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        results : dict
            Performances of the configuration on the instance set of the tournament.
        i : list
            List of next instances to run the tournament on.

        Returns
        -------
        ndarray
            **probimp**: Probabilities of improvement.
        """
        if len(results.values()) == 0:
            import sys
            self.best_val = sys.maxsize
        else:
            self.best_val = min(min(list(d.values()))
                                for d in list(results.values()))

        self.aafpi.update(eta=self.best_val, model=self.surr.model)

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

    def pca_inst_feat_file(self, train_insts, feature_file, insts, pca_feats):
        """Generate instance feature file with PCA features.

        Parameters
        ----------
        train_insts : str
            Name of instance file.
        feature_file : str
            path to file with problem instance features.
        insts : list of str
            Complete training instance set.
        pca_feats : ndarray
            Problem instance features.

        Returns
        -------
        tuple
            - ndarray,
              Instance features after PCA.
            - list,
              Instance names.

        """
        training_instances = []
        with open(f'{train_insts}', 'r') as f:
            for line in f:
                training_instances.append(line.strip())

        with open(f'{feature_file}', 'r') as f:
            feature_names = f.readline()

        self.selector_scenario.feature_file = \
            self.selector_scenario.log_location + self.selector_scenario.log_folder + \
            '/features_PCA.txt'

        with open(self.selector_scenario.feature_file, 'w') as f:
            f.write(feature_names)

        pca_inst_feats = []
        inst_ids = []

        with open(self.selector_scenario.feature_file, 'a') as f:
            for name, feats in zip(insts, pca_feats):
                if name in training_instances:
                    pca_inst_feats.append(feats)
                    inst_ids.append(name)
                    feat_string = ",".join(feats.astype('str'))
                    f.write(f"{name}, {feat_string}\n")

        return numpy.array(pca_inst_feats), inst_ids
