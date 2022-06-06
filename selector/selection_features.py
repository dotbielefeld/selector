"""This module contains feature generation functions."""
import numpy as np
from selector.pool import Generator


class FeatureGenerator:
    """Generate features necessary to evaluate configurations."""

    def __init__(self):
        """Initialize feature generation class."""
        self.Generator = Generator

    def percent_rel_evals(self, suggestions, data):
        """Percentage of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def avg_rel_evals_qual(self, suggestions, data):
        """Average quality of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def best_rel_evals_qual(self, suggestions, data):
        """Best target value relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def std_rel_evals_qual(self, suggestions, data):
        """Std of quality of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def diff_pred_real_qual(self, suggestions, data):
        """Difference of predicted and real quality of relatives evaluated so far.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def avg_dist_evals(self, suggestions, data):
        """Average distance to all pints so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def avg_dist_sel(self, suggestions, data):
        """Average distance to points in the current selection.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

    def avg_dist_rel(self, suggestions, data):
        """Average distance to relatives.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        return div_feats

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

    def dynamic_feature_gen(self, suggestions, data):
        """Generate static features.

        :param suggestions: list, suggested configurations
        :param data: data object, contains historic data
        :return dyn_feats: list, dynamic features
        """
        # TODO

        # Features based on GBM (Gradient Boosting Tree)
        dyn_feats.append(self.expected_improve(suggestions, data))
        dyn_feats.append(self.prob_qual_improve(suggestions, data, surr='GPR'))
        dyn_feats.append(self.prob_qual_improve(suggestions, data, surr='GPR'))
        dyn_feats.append(self.uncertainty_improve(suggestions, data))

        return np.array(dyn_feats)

    def diversity_feature_gen(self, suggestions, data):
        """Generate static features.

        :param suggestions: list, suggested configurations
        :param data: data object, contains historic data
        :return div_feats: list, diversity features
        """
        # TODO

        # Features based on relatives so far evaluated
        div_feats = self.percent_rel_evals(suggestions, data)
        div_feats.append(self.avg_rel_evals_qual(suggestions, data))
        div_feats.append(self.best_rel_evals_qual(suggestions, data))
        div_feats.append(self.std_rel_evals_qual(suggestions, data))
        div_feats.append(self.diff_pred_real_qual(suggestions, data))

        # Features based on all so far evaluated points
        div_feats.append(self.avg_dist_evals(suggestions, data))
        div_feats.append(self.avg_dist_sel(suggestions, data))
        div_feats.append(self.avg_dist_rel(suggestions, data))

        # TODO: One-Hot NeighborIneighbor values

        return np.array(div_feats)
