"""This module contains functions for the CPPL surrogate."""
import os
import copy
import sys
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
import uuid
import numpy as np
from scipy.linalg import sqrtm
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from selector.pool import Configuration, Generator, ParamType
from selector.point_gen import PointGen
from selector.generators.random_point_generator import random_point
from selector.generators.default_point_generator import (
    check_no_goods,
    check_conditionals,
    default_point
)
from selector.generators.random_point_generator import reset_conditionals

np.set_printoptions(threshold=sys.maxsize)
sys.path.append(os.getcwd())

from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()


class CPPL:
    """
    Surrogate from CPPL.

    Note
    ----

    Implementation based on the paper: "Pool-based realtime algorithm
    configuration: A preselection bandit approach"

    Parameters
    ----------
    s : selector.scenario.Scenario
        AC scenario.
    seed : int
        Random seed.
    features : ndarray
        Problem instance features.
    pool_size : int
        Number of Configurations to keep in CPPL pool.
    alpha : int
        CPPL hyperparameter alpha, see paper for details.
    gamma : float
        CPPL hyperparameter gamma, see paper for details.
    w : float
        CPPL hyperparameter w, see paper for details.
    random_prob : float
        Probability for a random Configuration being generated.
    mutation_prob : float
        Probability for mutation in crossover mechanism.
    pca_dimension_configurations : int
        PCA dimension for configuration values.
    pca_dimension_instances : int
        PCA dimension for instance feature values.
    model_update : str
        Update mode for model ["SGD", "Batch"].
    v_hat_norm : str
        Norm to use on v_hat [None, "max", "zero_one"], see paper for details.
    theta_norm : str
        Norm to use on theta [None, "max", "zero_one"], see paper for details.
    feature_normaliser : str
        Which normalization to use on instance feature values ["max", "zero_one"]. 
    ensemble: bool
        True, if accounting for same configuration with different IDs needs to be made.
    """

    def __init__(self, scenario, seed, features, pool_size=192, alpha=1,
                 gamma=0.1, w=0.1, random_prob=0.2, mutation_prob=0.8,
                 pca_dimension_configurations=8, pca_dimension_instances=8,
                 model_update="Batch", v_hat_norm=None, theta_norm="zero_one",
                 feature_normaliser="max", ensemble=True):
        
        self.scenario = scenario
        self.features = features
        self.seed = seed
        self.ensemble = ensemble

        self.v_hat_norm = v_hat_norm
        self.theta_norm = theta_norm
        self.feature_normaliser = feature_normaliser

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4())
                     for _ in range(self.pool_size - 1)] + \
                    [default_point(scenario, uuid.uuid4())]
        self.feature_store = {}

        self.model_update = model_update
        self.alpha = alpha
        self.gamma = gamma
        self.w = w
        self.t = 0

        self.random_prob = random_prob
        self.mutation_prob = mutation_prob
        self.number_new_confs = 3

        self.pca_dimension_configurations = pca_dimension_configurations
        if len(scenario.parameter) < self.pca_dimension_configurations:
            self.pca_dimension_configurations = len(scenario.parameter)

        self.pca_dimension_instances = pca_dimension_instances
        if len(list(self.features.values())[0]) < self.pca_dimension_instances:
            self.pca_dimension_instances = len(list(self.features.values())[0])
        elif len(self.features) < self.pca_dimension_instances:
            self.pca_dimension_instances = len(self.features)

        self.pca_number_confs_calibration = 100

        self.context_dim = self.pca_dimension_configurations * 2 + \
            self.pca_dimension_instances * 2 + \
            self.pca_dimension_configurations * self.pca_dimension_instances

        self.theta_hat = np.random.random_sample(self.context_dim)
        self.theta_bar = copy.copy(self.theta_hat)

        self.gradient_sum = {0: np.zeros((self.context_dim, self.context_dim))}
        self.hessian_sum = {0: np.zeros((self.context_dim, self.context_dim))}

        self.process_parameter()

        instance_feature_matrix = np.array(list(self.features.values()))
        self.instance_feature_standard_scaler = StandardScaler()
        transformed_features = \
            self.instance_feature_standard_scaler.\
            fit_transform(instance_feature_matrix)
        for instance, counter in zip(self.features.keys(),
                                     range(len(self.features.keys()))):
            self.features[instance] = transformed_features[counter]

        self.calibrate_pca()
        self.best_q = None
        self.identity_store = {}
        self.confidences = {}

    def process_parameter(self):
        """
        Figure out the parameter types and get values/calibrate encoding and scaling
        """

        cat_params = []
        cont_int_params = []
        # figure out the param type
        for param in self.scenario.parameter:
            if param.type == ParamType.categorical:
                cat_params.append(param)
            else:
                cont_int_params.append(param)

        # treat cont and int params
        self.lower_b_params = np.zeros(len(cont_int_params))
        self.upper_b_params = np.zeros(len(cont_int_params))

        for i in range(len(cont_int_params)):
            bound = cont_int_params[i].bound
            self.lower_b_params[i] = float(bound[0])
            self.upper_b_params[i] = float(bound[-1])

        # init one hot encoding for cat params:
        list_of_bounds = []
        list_o_of_default = []
        # With longest this is some super wired hack I have to do for the OneHotEncoder
        # If I do not find tha parameter with the longest string to parse in as default
        # the categories will become dtype Ux with x being the lenght of the longest string in the default
        # if then there are values that are longer then UX which get parsed in later these will be cut to the length..
        for param in cat_params:
            if isinstance(param.default, (bool, np.bool_)):
                list_of_bounds.append(list(map(str, list(map(int, param.bound)))))
                list_o_of_default.append(str(int(param.default)))
            else:
                longest = 0
                list_of_bounds.append(param.bound)
                for bound in param.bound:
                    if len(bound) > longest:
                        longest = len(bound)
                        cp = bound
                list_o_of_default.append(cp)

        self.cat_params_names = [p.name for p in cat_params]
        self.cont_int_params_names = [p.name for p in cont_int_params]

        if len(self.cat_params_names) > 0:
            self.o_h_enc = OneHotEncoder(categories=list_of_bounds)
            self.o_h_enc.fit(np.array(list_o_of_default).reshape(1, -1) )

    def scale_conf(self, configuration):
        """
        Scale and encode a configuration. Continuous/integer parameters are scaled between 0 and 1. Categorical parameters are one-hot encoded.

        Parameters
        ----------
        configuration : selector.pool.Configuration
            Configuration.

        Returns
        -------
        ndarray
            Scaled and encoded configuration values.
        """
        cat_params_on_conf = np.zeros(len(self.cat_params_names), dtype=object)
        cont_int_params_of_conf = np.zeros(len(self.cont_int_params_names), dtype=float)

        for param, value in configuration.conf.items():
            if param in self.cat_params_names:
                if isinstance(value, (bool, np.bool_)):
                    cat_params_on_conf[self.cat_params_names.index(param)] = str(int(value))
                else:
                    cat_params_on_conf[self.cat_params_names.index(param)] = value
            else:
                cont_int_params_of_conf[self.cont_int_params_names .index(param)] = value

        cont_int_scaled = (cont_int_params_of_conf - self.lower_b_params) / (self.upper_b_params - self.lower_b_params).reshape(1, -1)

        if len(self.cat_params_names) > 0:
            cat_params_on_conf = self.o_h_enc.transform(cat_params_on_conf.reshape(1, -1)).toarray()

        return np.concatenate((cont_int_scaled, cat_params_on_conf), axis=None)

    def calibrate_pca(self):
        """
        Calibarte the PCA and Scalers for the featuremap
        """

        random_generator = PointGen(self.scenario, random_point)
        if len(self.cat_params_names) > 0:
            para_vector_size = len([item for sublist in self.o_h_enc.categories_ for item in sublist]) + len(self.cont_int_params_names)
        else:
            para_vector_size = len(self.cont_int_params_names)

        conf_matrix = np.zeros((self.pca_number_confs_calibration, para_vector_size))
        for i in range(self.pca_number_confs_calibration):
            point = random_generator.point_generator()
            conf_matrix[i] = self.scale_conf(point)

        self.pca_configurations = PCA(n_components=self.pca_dimension_configurations)
        pca_conf = self.pca_configurations.fit_transform(conf_matrix)

        self.pca_instances = PCA(n_components=self.pca_dimension_instances)
        pca_features = self.pca_instances.fit_transform(np.array(list(self.features.values())))

        sample_size = min(pca_conf.shape[0], pca_features.shape[0])
        pca_conf = pca_conf[:sample_size]
        pca_features = pca_features[:sample_size]

        feature_map_matrix = np.zeros((sample_size * sample_size, self.context_dim))

        fc = 0
        for i in range(sample_size):
            for j in range(sample_size):
                feature_map_matrix[fc] = np.concatenate((pca_conf[i], pca_features[j], pca_conf[i] ** 2, pca_features[j] ** 2,
                                                         (pca_conf[i] * pca_features[j].reshape(-1,1)).flatten())).flatten()
                fc = fc + 1

        self.feature_map_scaler = StandardScaler()
        self.feature_map_scaler.fit(feature_map_matrix)

    def compute_feature_map(self, conf, instance_features):
        """
        For a configuration/instance pair, compute the quadratic feature map and scale.

        Parameters
        ----------
        conf : selector.pool.Configuration
            Configuration.
        instance_features : ndarray
            Numpy array of instance features.

        Returns
        -------
        ndarray
            Scaled quadratic problem instance features.
        """
        conf_values = self.scale_conf(conf)

        conf_values = self.pca_configurations.transform(conf_values.reshape(1, -1))
        instance_features = self.pca_instances.transform(instance_features.reshape(1, -1))

        features = np.concatenate((conf_values, instance_features, conf_values**2, instance_features ** 2,
                                   (conf_values * instance_features.reshape(-1,1)).flatten().reshape(1,-1)),axis=1).flatten()

        # features = features /max(features)
        if self.feature_normaliser == "max":
            features = features / max(features)
        elif self.feature_normaliser == "zero_one":
            features = (features - min(features)) / (max(features) - min(features ))

        return features

    def to_log_space(self):
        """Not implemented."""
        pass

    def from_log_space(self):
        """Not implemented."""
        pass

    def update_feature_store(self, conf, instance):
        """
        For a configuration/instance pair, compute the features and store them in a feature store for later use.

        Parameters
        ----------
        conf : selector.pool.Configuration
            Configuration.
        instance : str
            Instance name.
        """
        if conf.id not in self.feature_store:
            self.feature_store[conf.id] = {}

        if instance not in self.feature_store[conf.id]:
            self.feature_store[conf.id][instance] = self.compute_feature_map(conf, self.features[instance])

    def compute_a(self, theta, tried_conf_ids, instance_id):
        """
        Compute numerator for gradient and hessian computation.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        Returns
        -------
        ndarray
            Numerator for further compuations.
        """
        sum = 0

        for conf in tried_conf_ids:
            sum = sum + self.feature_store[conf][instance_id] * \
                np.exp(np.dot(self.feature_store[conf][instance_id], theta))
        return sum

    def compute_b(self, theta, tried_conf_ids, instance_id):
        """
        Compute denominator for gradient and hessian computation.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        Returns
        -------
        ndarray
            Denominator for further compuations.
        """
        sum = 0
        for conf in tried_conf_ids:
            sum = sum + np.exp(
                np.dot(self.feature_store[conf][instance_id], theta))
        return sum

    def compute_c(self, theta, tried_conf_ids, instance_id):
        """
        Compute second numerator for hessian computation.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        Returns
        -------
        ndarray
            Second numerator for further compuations.
        """
        sum = 0
        for conf in tried_conf_ids:
            sum = sum + np.exp(np.dot(self.feature_store[conf][instance_id], theta)) * \
                np.outer(self.feature_store[conf][instance_id], self.feature_store[conf][instance_id])
        return sum

    def compute_gradient(self, theta, winner_id, tried_confs, instance_id):
        """
        Compute the gradient for a given feedback.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        winner_id: uuid.UUID
            ID of the configuration winning the tournament.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        Returns
        -------
        ndarray
            Gradient for further compuations.
        """
        a = self.compute_a(theta, tried_confs, instance_id)
        b = self.compute_b(theta, tried_confs, instance_id)

        return self.feature_store[winner_id][instance_id] - (a / b)

    def compute_hessian(self, theta, winner_id, tried_confs, instance_id):
        """
        Compute the hessian for a given feedback.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        winner_id: uuid.UUID
            ID of the configuration winning the tournament.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        Returns
        -------
        ndarray
            Hessian for further compuations.
        """
        a = self.compute_a(theta, tried_confs, instance_id)
        b = self.compute_b(theta, tried_confs, instance_id)
        c = self.compute_c(theta, tried_confs, instance_id)
        return (np.outer(a, a) / b ** 2) - (c / b)

    def update_running_sums(self, theta, winner_id, tried_confs, instance_id):
        """
        Update the rolling sums of gradients and hessian for a feedback.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        winner_id: uuid.UUID
            ID of the configuration winning the tournament.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        """

        if self.t not in self.gradient_sum.keys():
            gradient = self.compute_gradient(theta, winner_id, tried_confs, instance_id)
            self.gradient_sum[self.t] = self.gradient_sum[self.t - 1] + np.outer(gradient, gradient)

        if self.t not in self.hessian_sum.keys():
            hessian = self.compute_hessian(theta, winner_id, tried_confs, instance_id)
            self.hessian_sum[self.t] = self.hessian_sum[self.t - 1] + hessian ** 2

    def compute_confidence(self, theta, instance_id, conf):
        """
        Compute the confidence for an instance/conf combination.

        Parameters
        ----------
        theta : ndarray
            Parameter vector theta_bar, based on winner feedback.
        instance_id : str
            Name of an instance.
        conf : uuid.UUID
            ID of a configuration.
        Returns
        -------
        float
            Confidence.
        """
        # Control How many threads/cores numpy and scipy use
        @controller.wrap(limits=self.scenario.tournament_size, user_api='openmp')
        @controller.wrap(limits=self.scenario.tournament_size, user_api='blas')
        def compute_confidence_threaded(theta, instance_id, conf):
            v = (1 / self.t) * self.gradient_sum[self.t]

            s = (1 / self.t) * self.hessian_sum[self.t]

            try:
                s_inv = np.linalg.inv(s)
            except:
                s_inv = np.linalg.pinv(s)

            sigma = (1 / self.t) * np.dot(np.dot(s_inv, v), s_inv)

            M = np.exp(2 * np.dot(self.feature_store[conf][instance_id], theta)) * \
                np.outer(self.feature_store[conf][instance_id], self.feature_store[conf][instance_id])

            sigma_root = sqrtm(sigma)
            I_hat = np.linalg.norm(np.dot(np.dot(sigma_root, M), sigma_root))

            return self.w * np.sqrt((2 * np.log(self.t) + self.context_dim + 2 * np.sqrt((self.context_dim * np.log(self.t)))) * I_hat)

        return compute_confidence_threaded(theta, instance_id, conf)

    def update_model_single_observation(self, winner_id, tried_confs, instance_id):
        """
        Update the thetas of the model for a single instance feedback.

        Parameters
        ----------
        winner_id: uuid.UUID
            ID of the configuration winning the tournament.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        """
        grad = self.compute_gradient(self.theta_hat, winner_id, tried_confs, instance_id)
        self.theta_hat = self.theta_hat + self.gamma * self.t ** (-self.alpha) * grad

        if self.theta_norm == "max":
            self.theta_hat = self.theta_hat / max(self.theta_hat)
        elif self.theta_norm == "zero_one":
            self.theta_hat = (self.theta_hat - min(self.theta_hat)) / (max(self.theta_hat ) - min(self.theta_hat ))
        #self.theta_hat = self.theta_hat/max(self.theta_hat)
        self.theta_bar = ((self.t - 1) * (self.theta_bar)) / self.t + (self.theta_hat / self.t)

    def update_model_mini_batch(self, winner_ids, tried_confs, instance_ids):
        """
        Update the thetas of the model for feedback over multiple instances.

        Parameters
        ----------
        winner_id: uuid.UUID
            ID of the configuration winning the tournament.
        tried_conf_ids : list of uuid.UUID
            IDs of configurations that participated in the tournament.
        instance_id : str
            Name of instance the feedback was generated on.
        """
        grad_mean = np.zeros(self.context_dim)

        for uc in range(len(winner_ids)):
            grad = self.compute_gradient(self.theta_hat, winner_ids[uc], tried_confs[uc], instance_ids[uc])
            grad_mean = grad_mean + grad

        grad_mean = grad_mean / len(winner_ids)
        self.theta_hat = self.theta_hat + self.gamma * self.t ** (-self.alpha) * grad_mean

        self.theta_hat = self.theta_hat / max(self.theta_hat)

        self.theta_bar = ((self.t - 1) * (self.theta_bar)) / self.t + (self.theta_hat / self.t)

    def select_from_set(self, conf_set, instance_set, n_to_select):
        """
        For a set of configurations and instances select the most promising configurations

        Parameters
        ----------
        conf_set : list of selector.pool.Configuration
            Set of configurations to select from.
        instance_set : list of str
            Instance set the next tournament will be run on.
        n_to_select : int
            Number of configurations to select from the set.
        Returns
        -------
        tuple
            - list of selector.pool.Configuration,
              Selected configurations.
            - list of list of float,
              Utility and confidence pairs according to selected configurations.
        """

        v_hat = np.zeros((len(conf_set), len(instance_set)))
        confidence = np.zeros((len(conf_set), len(instance_set)))
        for i, conf in enumerate(conf_set):
            for next_instance, _ in enumerate(instance_set):
                v_hat[i][next_instance] = np.exp(np.inner(self.feature_store[conf.id][instance_set[next_instance]], self.theta_bar))

                if self.t > 0:
                    confidence[i][next_instance] = self.compute_confidence(self.theta_bar, instance_set[next_instance], conf.id)

        v_hat_s = v_hat.sum(axis=1)

        if self.v_hat_norm == "max":
            v_hat_s = v_hat_s / max(v_hat_s)
        elif self.v_hat_norm == "zero_one":
            v_hat_s = (v_hat_s - min(v_hat_s)) / (max(v_hat_s) - min(v_hat_s))

        if self.t > 0:
            confidence_s = confidence.sum(axis=1)
            if self.v_hat_norm == "max":
                confidence_s = confidence_s / max(confidence_s)
            elif self.v_hat_norm == "zero_one":
                confidence_s = (confidence_s - min(confidence_s)) / (max(confidence_s) - min(confidence_s))
        else:
            confidence_s = np.zeros(v_hat_s.shape)

        quality = v_hat_s + confidence_s

        qual_sort = (-quality).argsort()

        selection = qual_sort[:n_to_select]
        return [conf_set[i] for i in selection], [[v_hat_s[i], confidence_s[i]] for i in qual_sort]

    def delete_from_pool(self, instance_set):
        """
        Based on the feedback delete poorly performing configurations from the pool.

        Parameters
        ----------
        instance_set : list of str
            Instance set the feedback was generated on.
        """
        v_hat = np.zeros((self.pool_size, len(instance_set)))
        confidence = np.zeros((self.pool_size, len(instance_set)))

        for i, conf in enumerate(self.pool):
            for next_instance, _ in enumerate(instance_set):
                v_hat[i][next_instance] = np.exp(np.inner(self.feature_store[conf.id][instance_set[next_instance]], self.theta_bar))

                if self.t > 0:
                    confidence[i][next_instance] = self.compute_confidence(self.theta_bar, instance_set[next_instance], conf.id)

        v_hat_s = v_hat.sum(axis=1)

        if self.v_hat_norm == "max":
            v_hat_s = v_hat_s / max(v_hat_s)
        elif self.v_hat_norm == "zero_one":
            v_hat_s = (v_hat_s - min(v_hat_s)) / (max(v_hat_s) - min(v_hat_s))

        if self.t > 0:
            confidence_s = confidence.sum(axis=1)
            if self.v_hat_norm == "max":
                confidence_s = confidence_s / max(confidence_s)
            elif self.v_hat_norm == "zero_one":
                confidence_s = (confidence_s - min(confidence_s)) / (max(confidence_s) - min(confidence_s))
        else:
            confidence_s = np.zeros(v_hat_s.shape)

        discard_index = []
        for c in range(self.pool_size):
            for oc in range(self.pool_size):
                if c != oc and v_hat_s[oc] - confidence_s[oc] > v_hat_s[c] + confidence_s[c]:
                    discard_index.append(c)
                    break
        dis = []
        for i in sorted(discard_index, reverse=True):
            dis.append(self.pool[i])
            del self.pool[i]

    def create_new_conf(self, parent_one, parent_two):
        """
        Create new configurations based on a genetic procedure described.

        Parameters
        ----------
        parent_one : selector.pool.Configuration
            First configuration for the crossover.
        parent_two : selector.pool.Configuration
            Second configuration for the crossover.
        Returns
        -------
        selector.pool.Configuration
            Resulting configuration.
        """
        no_good = True
        while no_good:
            new_conf = {}
            rn = np.random.uniform()

            if rn < self.random_prob:
                new_conf = random_point(self.scenario, uuid.uuid4())
                new_conf.generator = Generator.cppl
            else:
                for param, setting in parent_one.conf.items():

                    rn = np.random.uniform()
                    if rn < 0.5:
                        new_conf[param] = setting
                    else:
                        new_conf[param] = parent_two.conf[param]
                rn = np.random.uniform()

                if rn < self.mutation_prob:
                    possible_mutations = random_point(self.scenario, uuid.uuid4())
                    param_to_mutate = np.random.choice(list(new_conf.keys()))
                    new_conf[param_to_mutate] = possible_mutations.conf[param_to_mutate]

                identity = uuid.uuid4()

                new_conf = Configuration(identity,
                                         new_conf,
                                         Generator.cppl)

            cond_vio = check_conditionals(self.scenario, new_conf.conf)
            if cond_vio:
                new_conf.conf = reset_conditionals(self.scenario, new_conf.conf, cond_vio)

            no_good = check_no_goods(self.scenario, new_conf.conf)

        return new_conf

    def add_to_pool(self, past_instances):
        """
        Add the most promising newly created configurations to the pool

        Parameters
        ----------
        past_instances : list of str
            Instance set of the prior tournament.
        """
        number_to_create = self.pool_size - len(self.pool)
        new_promising_conf = []

        if number_to_create > 0:
            if len(self.pool) > 1:
                best_to_confs, _ = self.select_from_set(self.pool, past_instances, 2)
                conf_one, conf_two = best_to_confs[0], best_to_confs[1]
            else:
                conf_one = self.pool[0]
                conf_two = random_point(self.scenario, uuid.uuid4())
                conf_two.generator = Generator.cppl
            possible_new_confs = []
            for nc in range(self.number_new_confs * number_to_create):
                possible_new_confs = possible_new_confs + [self.create_new_conf(conf_one, conf_two)]
            for instance in past_instances:
                for c in possible_new_confs:
                    self.update_feature_store(c, instance)
            # TODO I have to ensure that all the confs here are diffrent...
            new_promising_conf, _ = self.select_from_set(possible_new_confs, past_instances, number_to_create)

        self.pool = self.pool + new_promising_conf

    def update(self, previous_tournament, c, results, terminations, instance_features=None, ac_runtime=None):
        """
        Update the model with given feedback.

        Parameters
        ----------
        results : dict of dict
            Feedback for the configuration-instance pairs in the previous tournament.
        previous_tournament : selector.pool.Tournament
            Tournament.
        c : list of selector.pool.Configuration
            Configurations participating in the previous tournament.
        results : dict
            Performances of the configuration on the instance set of the tournament.
        terminations : dict
            Configurations that were terminated on instances.
        instance_features : ndarray
            Problem instance features.
        ac_runtime : int
            The total runtime of Selector in seconds so far.
        """

        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        best_conf_store = []
        rest_conf_store = []
        instance_store = []

        confs_w_feedback = previous_tournament.best_finisher + previous_tournament.worst_finisher

        for instance in previous_tournament.instance_set:
            results_on_instance = {}

            for c in self.pool + confs_w_feedback:
                cond_vio = check_conditionals(self.scenario, c.conf)
                if cond_vio:
                    c.conf = reset_conditionals(self.scenario, c.conf, cond_vio)

                self.update_feature_store(c, instance)

            for c in previous_tournament.configuration_ids:
                if not np.isnan(results[c][instance]):
                    results_on_instance[c] = results[c][instance]
                else:
                    # OMIT the capped data in model update
                    if c in terminations:
                        if instance in terminations[c]:
                            continue
                    else:
                        # This conf/inst pair was a time limit reach
                        results_on_instance[c] = self.scenario.cutoff_time

            if self.best_q is None:
                if len(results_on_instance.values()) == 0:
                    import sys
                    self.best_q = sys.maxsize
                else:
                    self.best_q = min(results_on_instance.values())
            elif len(results_on_instance.values()) == 0: 
                if min(results_on_instance.values()) < self.best_q:
                    self.best_q = min(results_on_instance.values())

            best_conf_on_instance = min(results_on_instance, key=results_on_instance.get)

            if not self.ensemble:
                if results_on_instance[best_conf_on_instance] >= self.scenario.cutoff_time:
                    self.pool = [random_point(self.scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(self.scenario, uuid.uuid4())]
                    for c in self.pool:
                        [self.update_feature_store(c, i) for i in previous_tournament.instance_set]
                    continue

            tried = previous_tournament.configuration_ids.copy()

            # Through suggesting a conf multiple times with diffrent id we may get feedback for the same conf
            # under a diffrent id. Here we map that back in order to update for the same conf.
            if self.ensemble:
                if best_conf_on_instance in self.identity_store:
                    best_conf_on_instance = self.identity_store[best_conf_on_instance]
                    for c in range(len(tried)):
                        if tried[c] in self.identity_store:
                            tried[c] = self.identity_store[tried[c]]

            instance_store.append(instance)
            best_conf_store.append(best_conf_on_instance)
            rest_conf_store.append(tried)

            self.t = self.t + 1

            # this is actually the same as running "batch" with a best_conf_store of len() 1
            if self.model_update == "SGD":
                self.update_model_single_observation(best_conf_on_instance, tried, instance)

            self.update_running_sums(self.theta_bar, best_conf_on_instance, tried, instance)

        if self.model_update == "Batch" and len(best_conf_store) > 0:
            self.update_model_mini_batch(best_conf_store, rest_conf_store, instance_store)

        self.delete_from_pool(previous_tournament.instance_set)

        self.add_to_pool(previous_tournament.instance_set)

    def get_suggestions(self, scenario, n_to_select, d, r, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next based on the next instance set to run on.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        n_to_select : int
            Number of configurations to return.
        d : list of selector.pool.Tournament
            Tournament history.
        r : dict
            Performances of the configuration on the instance set of the tournament.
        next_instance_set : list of str
            Instance set to run on in the next tournament.
        instance_features : ndarray
            Problem instance features.
        Returns
        -------
        tuple
            - list of selector.pool.Configuration,
              Suggested configurations.
            - list of list of float,
              Utility and confidence pairs according to selected configurations.
        """

        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        for instance in next_instance_set:
            for c in self.pool:
                self.update_feature_store(c, instance)

        suggest, ranking = self.select_from_set(self.pool, next_instance_set, n_to_select)

        suggest = copy.deepcopy(suggest)
        for sugg in suggest:
            sugg.generator = Generator.cppl
            # We have to make sure that the ids of the configurations returend are unique.
            # I.e for two different tournaments we may suggest the same conf twice.
            # In that case the conf needs a unique id.
            # Later update() we need to map that back
            if self.ensemble:
                identity = uuid.uuid4()
                self.identity_store[identity] = sugg.id
                sugg.id = identity

        return suggest, ranking

    def suggest_from_outside_pool(self, conf_set, n_to_select, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next that are not in the CPPL pool.

        Parameters
        ----------
        conf_set : list of selector.pool.Configuration
        n_to_select : int
            Number of configurations to return.
        next_instance_set : list of str
            Instance set to run on in the next tournament.
        instance_features : ndarray
            Problem instance features.
        Returns
        -------
        tuple
            - **suggest**: list of selector.pool.Configuration,
              Suggested configurations.
            - **ranking**: list of list of float,
              Utility and confidence pairs according to selected configurations.
        """

        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        for instance in next_instance_set:
            for c in conf_set:
                self.update_feature_store(c, instance)

        suggest, ranking = self.select_from_set(conf_set, next_instance_set, n_to_select)

        return suggest, ranking

    def predict(self, suggestions, next_instance_set):
        """
        Predict performance/quality of configurations with CPPL.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        next_instance_set : 
            List of next instances to run the tournament on.
        Returns
        -------
        tuple
            - **v**: ndarray,
              Mean of predicted performance/quality.
            - **c**: ndarray,
              Variance of predicted performance/quality.
        """

        ranking = self.suggest_from_outside_pool(suggestions, len(suggestions),
                                                 next_instance_set)[1]
        v = []
        c = []

        for i, _ in enumerate(ranking):
            v.append(ranking[i][0])
            c.append(ranking[i][1])

        v = np.array(v)
        c = np.array(c)

        self.v = v
        self.c = c

        return v, c

    def probability_improvement(self, suggestions, results, next_instance_set):
        """
        Compute probability of improvement.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        next_instance_set : 
            List of next instances to be run.
        results : dict
            Performances of the configuration on the instance set of the tournament.
        next_instance_set : 
            List of next instances to run the tournament on.

        Returns
        -------
        ndarray
            **pi_output**: Probabilities of improvement.
        """

        v = self.v
        c = self.c

        std = np.sqrt(c)

        pi = norm.cdf((self.best_q - v) / std)
        pi_output = []
        for p in pi:
            pi_output.append([p])

        return pi_output

    def expected_improvement(self, suggestions, next_instance_set):
        """
        Compute expected improvement via CPPL model.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        next_instance_set : list of str
            List of next instances to be run.
        Returns
        -------
        ndarray
            **ei**: Expected improvements.
        """
        mean = self.v
        var = self.c

        std = np.sqrt(var)

        def calculate_ei():
            z = (self.best_q - mean) / std
            return (self.best_q - mean) * norm.cdf(z) + std * norm.pdf(z)

        if np.any(std == 0.0):
            stdc = np.copy(std)
            std[stdc == 0.0] = 1.0
            ei = calculate_ei()
            ei[stdc == 0.0] = 0.0
            return ei
        else:
            return calculate_ei()
