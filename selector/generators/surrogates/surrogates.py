"""This module contains surrogate management functions."""

import copy
from selector.pool import Surrogates
from selector.generators.surrogates.smac_surrogate import SmacSurr
from selector.generators.surrogates.ggapp_surrogate import GGAppSurr
from selector.generators.surrogates.cppl_surrogate import CPPL


class SurrogateManager():
    """
    Managing surrogates and related functions.

    Parameters
    ----------
    scenario : selector.scenario.Scenario
        AC scenario.
    seed : int
        Random seed.
    logger : logging.Logger
        Logger from main loop. Default is None, so no Debug infos.

    """

    def __init__(self, scenario, seed=False, logger=None):
        self.seed = seed
        self.surrogates = {
            Surrogates.SMAC: SmacSurr(scenario, seed=self.seed,
                                      pca_dim=scenario.smac_pca_dim),
            Surrogates.GGApp: GGAppSurr(scenario, seed=self.seed,
                                        logger=logger),
            Surrogates.CPPL: CPPL(scenario, seed=self.seed,
                                  features=scenario.features)
        }

    def suggest(self, suggestor, scenario, n_samples, data, results,
                next_instance_set):
        """Suggest points based on surrogate.

        Parameters
        ----------
        suggestor : selector.pool.Surrogates
            Key to the surrogate model.
        scenario : selector.scenario.Scenario
            AC scenario.
        seed : int
            Random seed.
        logger : logging.Logger
            Logger from main loop. Default is None, so no Debug infos.
        Returns
        -------
        list of selector.pool.Configuration
            Suggested configurations.
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

        Parameters
        ----------
        surrogate : selector.pool.Surrogates
            Key to the surrogate model.
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
        confs = copy.deepcopy(configs)
        self.surrogates[surrogate].update(history, confs, results,
                                          terminations,
                                          ac_runtime=ac_runtime)

    def predict(self, surrogate, configs, cot, next_instance_set):
        """
        Get prediction for mean and variance concerning the points quality.

        Parameters
        ----------
        surrogate : selector.pool.Surrogates
            Key to the surrogate model.
        configs : list of selector.pool.Configuration
            Suggested configurations.
        cot : float
            Timelimit set in AC scenario.
        next_instance_set : list 
            List of next instances to run the tournament on.
        Returns
        -------
        list of dict of dict
            [{"Configuartion ID": {"qual": mean predicted performance, "var": variance, "gen": selector.pool.Generator}}]
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
        """
        Compute expected improvement.

        Parameters
        ----------
        surrogate : selector.pool.Surrogates
            Key to the surrogate model.
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        next_instance_set : list of str
            List of next instances to be run.
        Returns
        -------
        ndarray or list
            **ei**: Expected improvements for suggestions.
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

        Parameters
        ----------
        surrogate : selector.pool.Surrogates
            Key to the surrogate model.
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        cot : float
            Timelimit set in AC scenario.
        results : dict
            Performances of the configuration on the instance set of the tournament.
        next_instance_set : list
            List of next instances to run the tournament on.

        Returns
        -------
        ndarray
            **pi**: Probabilities of improvement.
        """
        suggs = copy.deepcopy(suggestions)
        pi = self.surrogates[surrogate].\
            probability_improvement(suggs, results, next_instance_set)

        return pi
