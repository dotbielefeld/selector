"""This module contains feature generation functions."""
import numpy as np
from scipy.stats import norm
import copy
from selector.pool import Generator
import selector.hp_point_selection as hps


class FeatureGenerator:
    """Generate features necessary to evaluate configurations."""

    def __init__(self):
        """Initialize feature generation class."""
        self.Generator = Generator

    def percent_rel_evals(self, suggestions, data, nce):
        """Percentage of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        for sugg in suggestions:
            gen = sugg.generator
            rels = 0
            for content in data.values():
                for best in content.best_finisher:
                    if best.generator == gen:
                        rels += 1
                for worst in content.worst_finisher:
                    if worst.generator == gen:
                        rels += 1
            div_feats.append([rels / nce])

        return div_feats

    def avg_rel_evals_qual(self, suggestions, data, nce, results, cot,
                           generators):
        """Average quality of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :param nce: int, number of all configs evaluated
        :param results: dict, qualities of configurations
        :param cot: float, cut off time for tournaments
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        quals = {}
        counts = {}
        for gen in generators:
            for content in data.values():
                for best in content.best_finisher:
                    if best.generator == gen:
                        if gen in quals:
                            quals[gen] += sum(results[best.id].values()) / \
                                len(results[best.id])
                            counts[gen] += 1
                        else:
                            quals[gen] = sum(results[best.id].values()) / \
                                len(results[best.id])
                            counts[gen] = 1
                for worst in content.worst_finisher:
                    if worst.generator == gen:
                        if gen in quals:
                            quals[gen] += sum(results[worst.id].values()) / \
                                len(results[worst.id])
                            counts[gen] += 1
                        else:
                            quals[gen] = sum(results[worst.id].values()) / \
                                len(results[worst.id])
                            counts[gen] = 1

        for gen in quals.keys():
            quals[gen] = quals[gen] / counts[gen]

        for sugg in suggestions:
            if sugg.generator not in quals:
                div_feats.append([0])
            else:
                div_feats.append([quals[sugg.generator] / cot])

        return div_feats

    def best_rel_evals_qual(self, suggestions, data, generators, results, cot):
        """Best target value relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        best_val = {}
        for gen in generators:
            for content in data.values():
                for best in content.best_finisher:
                    if best.generator == gen:
                        for val in results[best.id].values():
                            if gen not in best_val:
                                best_val[gen] = val
                            elif val < best_val[gen]:
                                best_val[gen] = val
                for worst in content.worst_finisher:
                    if worst.generator == gen:
                        for val in results[worst.id].values():
                            if gen not in best_val:
                                best_val[gen] = val
                            elif val < best_val[gen]:
                                best_val[gen] = val
        for sugg in suggestions:
            if sugg.generator not in best_val:
                div_feats.append([0])
            else:
                div_feats.append([best_val[sugg.generator] / cot])

        return div_feats

    def std_rel_evals_qual(self, suggestions, data, generators, results, cot):
        """Std of quality of relatives so far evaluated.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        div_feats = []
        qual_vals = {}
        for gen in generators:
            for content in data.values():
                for best in content.best_finisher:
                    if best.generator == gen:
                        if gen in qual_vals:
                            for res_val in list(results[best.id].values()):
                                qual_vals[gen].append(res_val)
                        else:
                            qual_vals[gen] = list(results[best.id].values())
                for worst in content.worst_finisher:
                        if gen in qual_vals:
                            for res_val in list(results[worst.id].values()):
                                qual_vals[gen].append(res_val)
                        else:
                            qual_vals[gen] = list(results[worst.id].values())

        qual_std = {}

        for key, qv in qual_vals.items():
            qual_std[key] = np.std(qv)

        for sugg in suggestions:
            if sugg.generator not in qual_std:
                div_feats.append([0])
            else:
                div_feats.append([qual_std[sugg.generator] / cot])

        return div_feats

    def diff_pred_real_qual(self, suggestions, data, predicted_quals, results):
        """Difference of predicted and real quality of relatives evaluated so far.

        :param suggestions: list, suggested points
        :param data: data object, contains historic performance data
        :return div_feats: list, computed features of suggested points
        """
        if not predicted_quals:
            div_feats = [[0] for _ in suggestions]
        else:
            rel_results = {}
            rel_predicts = {}
            div_feats = []
            diffs = {}
            for sugg in suggestions:
                gen = sugg.generator
                for pred in predicted_quals:
                    pred = list(pred.values())[0]
                    if gen == pred['gen']:
                        if gen in rel_predicts:
                            rel_predicts[gen].append(pred['qual'])
                        else:
                            rel_predicts[gen] = [pred['qual']]
                for content in data.values():
                    for best in content.best_finisher:
                        if best.generator == gen:
                            if gen in rel_results:
                                for key, res in results[best.id].items():
                                    rel_results[gen].append(res)
                            else:
                                rel_results[gen] = \
                                    [list(results[best.id].values())[0]]
                                first = [list(results[best.id].keys())[0]]
                                for key, res in results[best.id].items():
                                    if key != first:
                                        rel_results[gen].append(res)
                    for worst in content.worst_finisher:
                        if worst.generator == gen:
                            if gen in rel_results:
                                for key, res in results[worst.id].items():
                                    rel_results[gen].append(res)
                            else:
                                rel_results[gen] = \
                                    [list(results[worst.id].values())[0]]
                                first = [list(results[worst.id].keys())[0]]
                                for key, res in results[worst.id].items():
                                    if key != first:
                                        rel_results[gen].append(res)

            for gen in Generator:
                if gen in rel_results and gen in rel_predicts:
                    if len(rel_predicts[gen]) > 0 and \
                            len(rel_results[gen]) > 0:
                        diffs[gen] = \
                            sum(rel_predicts[gen]) / len(rel_predicts[gen]) - \
                            sum(rel_results[gen]) / len(rel_results[gen])
                elif gen not in diffs:
                    diffs[gen] = 0

            for sugg in suggestions:
                div_feats.append([diffs[sugg.generator]])

        return div_feats

    def avg_dist_evals(self, suggests, evals, psetting):
        """Average distance to all points so far evaluated.

        :param suggestions: list, suggested points
        :param evals: list, already evaluated points
        :param psetting: scenario.parameter
        :return div_feats: list, average distances to all already evaluated
                                 points
        """
        if evals:

            suggestions = copy.deepcopy(suggests)
            evaluated = copy.deepcopy(evals)

            suggestions = hps.normalize_plus_cond_acc(suggestions, psetting)
            evaluated = hps.normalize_plus_cond_acc(evaluated, psetting)
            distances = hps.pairwise_distances(suggestions, evaluated)

            div_feats = []
            for dist in distances:
                div_feats.append([np.mean(dist)])

        else:
            div_feats = [[0] for _ in suggests]

        return div_feats

    def avg_dist_sel(self, suggests, psetting):
        """Average distance to points in the current selection.

        :param suggestions: list, suggested points
        :param psetting: scenario.parameter
        :return div_feats: list, average distances to points in current
                                 selection
        """
        suggestions = copy.deepcopy(suggests)

        suggestions = hps.normalize_plus_cond_acc(suggestions, psetting)
        distances = hps.pairwise_distances(suggestions, suggestions)

        div_feats = []
        for dist in distances:
            div_feats.append([np.mean(dist)])

        return div_feats

    def avg_dist_rel(self, suggests, evals, psetting, generators):
        """Average distances to relatives.

        :param suggests: list, suggested points
        :param evals: list, already evaluated points
        :param psetting: scenario.parameter
        :param generators: list, available generators
        :return div_feats: list, computed features of suggested points
        """
        if evals:
            suggestions = copy.deepcopy(suggests)
            evaluated = copy.deepcopy(evals)

            suggestions = hps.normalize_plus_cond_acc(suggestions, psetting)
            evaluated = hps.normalize_plus_cond_acc(evaluated, psetting)

            group_relatives = {}
            for gen in generators:
                for ev in evaluated:
                    if gen == ev.generator:
                        if gen not in group_relatives:
                            group_relatives[gen] = [ev]
                        else:
                            group_relatives[gen].append(ev)

            distances = []
            for sugg in suggestions:
                if sugg.generator in group_relatives:
                    distances.append(hps.pairwise_distances([sugg],
                                     group_relatives[sugg.generator]))
                else:
                    distances.append([0 for _ in sugg.conf])

            div_feats = []
            for dist in distances:
                div_feats.append([np.mean(dist)])

        else:
            div_feats = [[0] for _ in suggests]

        return div_feats

    def expected_qual(self, suggests, sm, cot, psetting, surr):
        """Expected quality of points.

        :param suggests: list, suggested points
        :param sm: initialized surrogates.SurrogateManager()
        :param cot: int, cut off time
        :param psetting: scenario.parameter
        :param surr: which surrogate to use
        :return dyn_feats: list, computed features of suggested points
        """
        dyn_feats = []
        try:
            expimp = sm.expected_value(suggests, psetting, cot,
                                       surrogate=surr)

            for exim in expimp:
                for ei in exim.values():
                    dyn_feats.append([ei['qual']])

        except:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def prob_qual_improve(self, suggests, sm, cot, psetting, results, surr):
        """Probability of quality of points to improve.

        :param suggests: list, suggested points
        :param sm: initialized surrogates.SurrogateManager()
        :param cot: int, cut off time
        :param psetting: scenario.parameter
        :param surr: which surrogate to use
        :param results: list, results of points evaluated so far
        :return dyn_feats: list, computed features of suggested points
        """
        uncertainty = sm.uncertainty(suggests, psetting, surrogate=surr)
        if not all(v == 0 for v in uncertainty):
            dyn_feats = []
            result_quals = []
            for res in results.values():
                for p, r in res.items():
                    result_quals.append(r)

            expimp = sm.expected_value(suggests, psetting, cot,
                                       surrogate=surr)
            expected_im = []
            for exim in expimp:
                for ei in exim.values():
                    expected_im.append(ei['qual'])

            probim = norm(expected_im, uncertainty).cdf(np.min(result_quals))
            for pe in probim:
                dyn_feats.append([pe])
        else:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def uncertainty_improve(self, suggests, psetting, sm, surr):
        """Probability of quality of points to improve.

        :param suggests: list, suggested points
        :param sm: initialized surrogates.SurrogateManager()
        :param cot: int, cut off time
        :param psetting: scenario.parameter
        :param surr: which surrogate to use
        :param results: list, results of points evaluated so far
        :return dyn_feats: list, computed features of suggested points
        """
        uncertainty = sm.uncertainty(suggests, psetting, surrogate=surr)
        if not all(v == 0 for v in uncertainty):
            dyn_feats = [[uncert] for uncert in uncertainty]
        else:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

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

    def dynamic_feature_gen(self, suggestions, data, predicted_quals, sm,
                            cot, psetting, results):
        """Generate dynamic features.

        :param suggestions: list, suggested configurations
        :param data: data object, contains historic data
        :return dyn_feats: list, dynamic features
        """
        # TODO
        dyn_feats = []

        # Features based on GBM (Gradient Boosting Tree)
        '''
        dyn_feats = self.expected_qual(suggestions, sm,
                                       cot, psetting, surr='SMAC')
        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.expected_improve(suggestions,
                                                  predicted_quals,
                                                  surr='GGA')),
                           axis=1)
        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.prob_qual_improve(suggestions,
                                                   data, surr='SMAC')),
                           axis=1)
         dyn_feats = \
            np.concatenate((dyn_feats,
                            self.prob_qual_improve(suggestions, data,
                                                   surr='GGA')),
                           axis=1)
         dyn_feats = \
            np.concatenate((dyn_feats,
                            self.uncertainty_improve(suggestions, data,
                                                     surr='SMAC')),
                           axis=1)
         dyn_feats = \
            np.concatenate((dyn_feats,
                            self.uncertainty_improve(suggestions, data,
                                                     surr='GGA')),
                           axis=1)
        '''

        return np.array(dyn_feats)

    def diversity_feature_gen(self, suggestions, data, results, cot,
                              psetting, predicted_quals, evaluated):
        """Generate diversity features.

        :param suggestions: list, suggested configurations
        :param data: data object, contains historic data
        :param results: dict, qualities of configurations
        :param cot: float, cut off time for tournaments
        :param psetting: scenario.parameter
        :param predicted_quals: list, predicted qualities of
                                all points evaluated so far
        :param evaluated: list, all evaluated points so far
        :param sm: initialized surrogates.SurrogateManager()
        :return div_feats: list, diversity features
        """
        nce = 0
        for content in data.values():
            nce += len(content.configuration_ids)

        generators = [gen for gen in Generator]

        # Features based on relatives evaluated so far
        div_feats = self.percent_rel_evals(suggestions, data, nce)
        div_feats = \
            np.concatenate((div_feats,
                            self.avg_rel_evals_qual(suggestions, data,
                                                    nce, results, cot,
                                                    generators)),
                           axis=1)
        div_feats = \
            np.concatenate((div_feats,
                            self.best_rel_evals_qual(suggestions, data,
                                                     generators, results,
                                                     cot)),
                           axis=1)
        div_feats = \
            np.concatenate((div_feats,
                            self.std_rel_evals_qual(suggestions, data,
                                                    generators, results,
                                                    cot)),
                           axis=1)
        '''
        div_feats = \
            np.concatenate((div_feats,
                           self.diff_pred_real_qual(suggestions, data,
                                                    predicted_quals,
                                                    results)),
                           axis=1)
        '''

        # Features based on points evaluated so far
        div_feats = \
            np.concatenate((div_feats,
                           self.avg_dist_evals(suggestions, evaluated,
                                               psetting)),
                           axis=1)
        div_feats = \
            np.concatenate((div_feats,
                           self.avg_dist_sel(suggestions, psetting)),
                           axis=1)
        div_feats = \
            np.concatenate((div_feats,
                           self.avg_dist_rel(suggestions, evaluated,
                                             psetting, generators)),
                           axis=1)

        return np.array(div_feats)
