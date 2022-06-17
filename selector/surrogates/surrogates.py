"""This module contains surrogate management functions."""
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real, Space
from bayesmark.abstract_optimizer import AbstractOptimizer

from selector.pool import ParamType


class SurrogateManager(AbstractOptimizer):
    """Managing surrogates and related functions."""

    primary_import = "scikit-optimize"

    def __init__(self, psetting, n_initial_points=5):
        """Initialize surrogate managing class."""
        self.psetting = psetting
        dimensions = self.transform_param_space_skopt(psetting)
        self.space = Space(dimensions)
        self.surrogates = {'GPR': Optimizer(dimensions,
                                            n_initial_points=n_initial_points,
                                            base_estimator='GP', acq_func='PI',
                                            acq_optimizer="sampling")}

    def transform_param_space_skopt(self, psetting, transform="normalize"):
        """Transform parameter setting for skopt search space.

        :param s: scenario
        :param transorm: str, skopt parameter
        :return dimensions: list, transformed parameter setting
        """
        # Sort to avoid potential problems with space.py
        parameters = {}
        for p in psetting:
            parameters[p.name] = {'type': p.type, 'bound': p.bound,
                                  'scale': p.scale}

        parameters = dict(sorted(parameters.items()))
        dimensions = []
        for key, param in parameters.items():
            if param['type'] == ParamType.integer:
                dimensions.append(Integer(param['bound'][0],
                                          param['bound'][-1],
                                          transform=transform,
                                          name=key))
            elif param['type'] == ParamType.categorical:
                if len(param['bound']) > 2:
                    dimensions.append(Categorical(param['bound'],
                                                  name=key))
                else:
                    dimensions.append(Integer(0, 1,
                                      transform=transform,
                                      name=key))
            elif param['type'] == ParamType.continuous:
                pr = "log-uniform" if param['scale'] in ("l", "logit") \
                    else "uniform"
                dimensions.append(Real(param['bound'][0],
                                       param['bound'][-1],
                                       prior=pr,
                                       transform=transform,
                                       name=key))

        return dimensions

    def transform_config(self, config, psetting):
        """Transform configuration for skopt search space.

        :param config: dict, configuration
        :param psetting: scenario.parameter
        :return config: list, transformed config
        """
        for param in psetting:
            if param.name in config:
                if param.type == ParamType.categorical and \
                        len(param.bound) == 2:
                    if config[param.name] is True:
                        config[param.name] = 0
                    else:
                        config[param.name] = 1
            else:
                config[param.name] = param.default

        config = list(dict(sorted(config.items())).values())

        return config

    def expected_value(self, suggestions, psetting, cot, surrogate):
        """Compute expected configuration quality.

        :param suggestions: selecte configurationsy
        :param psetting: scenario.parameter
        :param cot: float, cutoff time
        :param surrogate: str, surrogate to be updated
        :return: list, expected configuration qualities
        """
        try:
            tr_suggs = []
            for sugg in suggestions:
                tr_suggs.append(self.transform_config(sugg.conf, psetting))

            quals = \
                self.surrogates[surrogate].models[-1].predict(
                    self.space.transform(tr_suggs), return_std=False)
            return [{sugg.id: {'qual': quals[s], 'gen': sugg.generator}}
                    for s, sugg in enumerate(suggestions)]
        except:
            return [{sugg.id: {'qual': cot, 'gen': sugg.generator}}
                    for sugg in suggestions]

    def uncertainty(self, suggestions, psetting, surrogate):
        """Compute uncertainty of predicted quality.

        :param suggestions: list, transformed suggested points values
        :param surr: which surrogate to use
        :return ysd: uncertainties of predictions of point qualities

        """
        try:
            tr_suggs = []
            for sugg in suggestions:
                tr_suggs.append(self.transform_config(sugg.conf, psetting))

            _, ysd = \
                self.surrogates[surrogate].models[-1].predict(
                    self.space.transform(tr_suggs), return_std=True)
            return ysd
        except:

            return [0 for _ in suggestions]

    def observe(self, point, result, psetting, cot, surrogate):
        """Update surrogate with observed configuration quality.

        :param point: Configuration object, evaluated point
        :param result: float or int, observed quality
        :param psetting: scenario.parameter
        :param cot: float, cutoff time
        :param surrogate: str, surrogate to be updated
        """
        point = self.transform_config(point, psetting)

        if result != cot:
            self.surrogates[surrogate].tell(point, result)

    def suggest(self, n_suggestions, suggestor='LHS', surrogate='GBM',
                suggestions=None, suggestor_options=None):
        """Let a surrogate suggest configurations.

        :param n_suggestions: int, number of points to suggest
        :param suggestor: str, suggestor name
        :param surrogate: str, surrogate name
        :param suggestions: list, suggestions from other sources
        :param suggestor_options: options for the suggestor
        :return suggestion: configuration(s) suggested
        """
        # TODO
        suggestion = 0

        return suggestion
