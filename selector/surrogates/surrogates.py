"""This module contains surrogate management functions."""

from selector.pool import Surrogates
from selector.surrogates.smac_surrogate import SmacSurr


class SurrogateManager():
    """Managing surrogates and related functions."""

    def __init__(self, scenario, seed=False):
        """Initialize surrogate managing class.

        :param scenario: object, selector.scenario
        :param seed: int, random seed
        """
        self.seed = seed
        self.surrogates = {
            Surrogates.SMAC: SmacSurr(scenario, seed=self.seed)
        }

    def suggest(self, suggestor, scenario, n_samples=1):
        """Suggest points based on surrogate.

        :param suggestor: object Surrogates, which surrogate to use
        :param scenario: object, selector.scenario
        :param n_samples: int, how many points to suggest
        :return sugg: list, suggested points
        """
        sugg = self.surrogates[suggestor].get_suggestions(scenario, n_samples)

        return sugg

    def update_surr(self, surrogate, history, conf, state, tourn_nr):
        """Update surrogate model with runhistory.

        :param surrogate: object Surrogates, which surrogate to use
        :param history: Tournament history
        :param conf: list, configurations, which history to update with
        :param state: object selector.pool.Status, status of this point
        :param tourn_nr: int, number of tournament, which to update with
        """
        self.surrogates[surrogate].update(history, conf, state, tourn_nr)

    def predict(self, surrogate, suggestions, cot):
        """Get prediction for mean and variance concerning the points quality.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :param cot: float, cut off time for tournaments
        :return predictions: list of dicts, contains info and predictions
            for regarded configurations
        """
        try:
            predict = self.surrogates[surrogate].predict(suggestions)
            mean = predict[0]
            var = predict[1]
            return [{sugg.id: {'qual': mean[s][0], 'var': var[s][0],
                               'gen': sugg.generator}}
                    for s, sugg in enumerate(suggestions)]
        except:
            return [{sugg.id: {'qual': cot, 'var': 0,
                               'gen': sugg.generator}}
                    for sugg in suggestions]

    def ei(self, surrogate, suggestions):
        """Compute expected improvement.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :return ei: nested list, expected improvements
        """
        try:
            ei = self.surrogates[surrogate].expected_improvement(suggestions)
            return ei
        except:
            return [[0] for sugg in suggestions]

    def pi(self, surrogate, suggestions, cot):
        """Compute probability of improvement.

        :param surrogate: object Surrogates, which surrogate to use
        :param suggestions: list, suggested configurations
        :param cot: float, cut off time for tournaments
        :return pi: nested list, probabilities of improvement
        """
        pi = self.surrogates[surrogate].probability_improvement(suggestions)

        return pi
