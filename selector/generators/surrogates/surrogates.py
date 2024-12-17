"""This module contains surrogate management functions."""

import copy
from selector.pool import Surrogates
from selector.generators.surrogates.smac_surrogate import SmacSurr
from selector.generators.surrogates.ggapp_surrogate import GGAppSurr
from selector.generators.surrogates.cppl_surrogate import CPPL


class SurrogateManager():
    """Managing surrogates and related functions."""

    def __init__(self, scenario, seed=False, logger=None):
        """Initialize surrogate managing class.

        :param scenario: object, selector.scenario
        :param seed: int, random seed
        """
        self.seed = seed
        self.surrogates = {
            Surrogates.SMAC: SmacSurr(scenario, seed=self.seed,
                                      pca_dim=scenario.smac_pca_dim),
            Surrogates.GGApp: GGAppSurr(scenario, seed=self.seed, logger=logger),
            Surrogates.CPPL: CPPL(scenario, seed=self.seed,
                                  features=scenario.features)
        }

    def suggest(self, suggestor, scenario, n_samples, data, results,
                next_instance_set):
        """Suggest points based on surrogate.

        :param suggestor: object Surrogates, which surrogate to use
        :param scenario: object, selector.scenario
        :param n_samples: int, how many points to suggest
        :return sugg: list, suggested points
        """
        sugg = \
            self.surrogates[suggestor].get_suggestions(scenario,
                                                       n_samples,
                                                       data, results,
                                                       next_instance_set)

        return sugg

    def update_surr(self, surrogate, history, configs, results, terminations,
                    ac_runtime=None):
        """Update surrogate model with runhistory.

        :param surrogate: object Surrogates, which surrogate to use
        :param history: Tournament history
        :param conf: list, configurations, which history to update with
        :param state: object selector.pool.Status, status of this point
        :param tourn_nr: int, number of tournament, which to update with
        """
        confs = copy.deepcopy(configs)
        self.surrogates[surrogate].update(history, confs, results,
                                          terminations,
                                          ac_runtime=ac_runtime)

    def predict(self, surrogate, configs, cot, next_instance_set):
        """Get prediction for mean and variance concerning the points quality.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :param cot: float, cut off time for tournaments
        :return predictions: list of dicts, contains info and predictions
            for regarded configurations
        """
        suggestions = copy.deepcopy(configs)
        if surrogate == Surrogates.SMAC:
            predict = self.surrogates[surrogate].predict(suggestions,
                                                         next_instance_set)

        try:
            predict = self.surrogates[surrogate].predict(suggestions,
                                                         next_instance_set)

            mean = predict[0]
            var = predict[1]

            return [{sugg.id: {'qual': mean[s], 'var': var[s],
                               'gen': sugg.generator}}
                    for s, sugg in enumerate(suggestions)]
        except:
            return [{sugg.id: {'qual': cot, 'var': 0,
                               'gen': sugg.generator}}
                    for sugg in suggestions]

    def ei(self, surrogate, suggestions, next_instance_set):
        """Compute expected improvement.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :return ei: nested list, expected improvements
        """
        suggs = copy.deepcopy(suggestions)
        try:
            ei = self.surrogates[surrogate].\
                expected_improvement(suggs, next_instance_set)

            return ei
        except:
            return [[0] for sugg in suggestions]

    def pi(self, surrogate, suggestions, cot, results, next_instance_set):
        """Compute probability of improvement.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :param cot: float, cut off time for tournaments
        :return pi: nested list, probabilities of improvement
        """
        suggs = copy.deepcopy(suggestions)
        pi = self.surrogates[surrogate].\
            probability_improvement(suggs, results, next_instance_set)

        return pi
