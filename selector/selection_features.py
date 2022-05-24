"""This module contains feature generation functions."""
import numpy as np
from selector.pool import Generator


class FeatureGenerator:
    """Generate features necessary to evaluate configurations."""

    def __init__(self):
        """Initialize feature generation class."""
        self.Generator = Generator

    def static_feature_gen(self, suggestions, epoch, max_epoch):
        """Generate static features.

        :param suggestions: list, suggested configurations
        :param epoch: int, current epoch
        :param max_epoch: int, total number of epochs
        :return static_features: list, static features
        """
        static_feats = [[] for ii in range(len(suggestions))]

        # One-Hot encoded information of generator used for conf
        for s in range(len(suggestions)):
            for gt in range(len(self.Generator)):
                if suggestions[s].generator == self.Generator(gt + 1):
                    static_feats[s].append(1)
                else:
                    static_feats[s].append(0)

        # Ratio of current epoch and max. epochs
        for sf in range(len(static_feats)):
            static_feats[sf].append(epoch / max_epoch)

        return np.array(static_feats)

    def dynamic_feature_gen(self, suggestions):
        """Generate static features.

        :param suggestions: list, suggested configurations
        :return dynamic_features: list, dynamic features
        """
        # TODO

        return np.array([])
