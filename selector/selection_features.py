"""This module contains feature generation functions."""
import numpy as np
import copy
import time
from collections import defaultdict
from selector.pool import Generator, Surrogates
import selector.hp_point_selection as hps


class FeatureGenerator:
    """
    Generate features necessary to evaluate configurations.

    Parameters
    ----------
    logger : logging.Logger
        Logging object.
    """

    def __init__(self, logger=None):
        self.Generator = Generator
        self.logger = logger

    def percent_rel_evals(self, suggestions, data, nce):
        """Percentage of relatives so far evaluated.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        data : object
            Contains historic performance data.
        nce : int
            Number of configuration evaluations.

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
        """
        self.gen_counts = defaultdict(int)

        for content in data.values():
            for finisher in content.best_finisher + content.worst_finisher:
                self.gen_counts[finisher.generator] += 1

        div_feats = [[self.gen_counts[sugg.generator]]
                     for sugg in suggestions]

        max_val = max(max(d) for d in div_feats)
        if max_val != 0.0:
            for d in div_feats:
                d[0] = d[0] / max_val

        return div_feats

    def avg_rel_evals_qual(self, suggestions, data, nce, results, cot,
                           generators):
        """Average quality of relatives so far evaluated.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        data : object
            Contains historic performance data.
        nce : int
            Number of all configs evaluated.
        results : dict
            Qualities of configurations.
        cot : float
            Cut off time for tournaments (i.e. time limit).
        generators : list
            All possible generators.

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
        """
        div_feats = []
        quals = {}
        gen_counts = self.gen_counts
        quals = defaultdict(int)
        for content in data.values():
            for finisher in content.best_finisher + content.worst_finisher:
                quals[finisher.generator] += \
                    sum(results[finisher.id].values()) / \
                    len(results[finisher.id])

        for gen in quals.keys():
            quals[gen] /= gen_counts.get(gen, 1)  # Avoid division by zero

        div_feats = [[quals.get(sugg.generator, 0) / cot]
                     for sugg in suggestions]

        max_val = max(max(d) for d in div_feats)
        if max_val != 0.0:
            for d in div_feats:
                d[0] /= max_val

        return div_feats

    def best_rel_evals_qual(self, suggestions, data, generators, results, cot):
        """Best target value relatives so far evaluated.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        data : object
            Contains historic performance data.
        generators : list
            All possible generators.
        results : dict
            Qualities of configurations.
        cot : float
            Cut off time for tournaments (i.e. time limit).

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
        """
        div_feats = []
        best_val = {}
        best_val = defaultdict(int)
        for content in data.values():
            for finisher in content.best_finisher + content.worst_finisher:
                for val in results[finisher.id].values():
                    if finisher.generator not in best_val:
                        best_val[finisher.generator] = val
                    elif val < best_val[finisher.generator]:
                        best_val[finisher.generator] = val

        div_feats = [[best_val.get(sugg.generator, 0) / cot]
                     for sugg in suggestions]

        max_val = max(max(d) for d in div_feats)
        if max_val != 0.0:
            for d in div_feats:
                d[0] = d[0] / max_val

        return div_feats

    def std_rel_evals_qual(self, suggestions, data, generators, results, cot):
        """Std of quality of relatives so far evaluated.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        data : object
            Contains historic performance data.
        generators : list
            All possible generators.
        results : dict
            Qualities of configurations.
        cot : float
            Cut off time for tournaments (i.e. time limit).

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
        """
        div_feats = []
        qual_vals = {}
        qual_vals = defaultdict(int)
        for content in data.values():
            for finisher in content.best_finisher + content.worst_finisher:
                if finisher.generator in qual_vals:
                    for res_val in list(results[finisher.id].values()):
                        qual_vals[finisher.generator].append(res_val)
                else:
                    qual_vals[finisher.generator] = \
                        list(results[finisher.id].values())

        qual_std = {}

        for key, qv in qual_vals.items():
            qual_std[key] = np.std(qv)

        div_feats = [[qual_std.get(sugg.generator, 0) / cot]
                     for sugg in suggestions]

        max_val = max(max(d) for d in div_feats)
        if max_val != 0.0:
            for d in div_feats:
                d[0] = d[0] / max_val

        return div_feats

    def diff_pred_real_qual(self, suggestions, data, predicted_quals, results):
        """Difference of predicted & real qual. of relatives evaluated so far.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        data : object
            Contains historic performance data.
        predicted_quals : list of lists
            Predicted performance/quality for suggested configurations.
        results : dict
            Qualities of configurations.

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
        """
        if not predicted_quals:
            div_feats = [[0] for _ in suggestions]
        else:
            rel_results = {}
            rel_predicts = {}
            div_feats = []
            diffs = {}

            # Collect relevant predictions into a dictionary
            for pred in predicted_quals:
                pred = list(pred.values())[0]
                gen = pred['gen']
                rel_predicts.setdefault(gen, []).append(pred['qual'])

            # Collect relevant results into a dictionary
            for content in data.values():
                for finisher in content.best_finisher + content.worst_finisher:
                    gen = finisher.generator
                    rel_results.setdefault(gen, []).extend(results[finisher.id].values())

            for gen in Generator:
                if gen in rel_results and gen in rel_predicts:
                    if len(rel_predicts[gen]) > 0 and \
                            len(rel_results[gen]) > 0:
                        diffs[gen] = \
                            np.mean(rel_predicts[gen]) \
                            / np.mean(rel_results[gen])
                elif gen not in diffs:
                    diffs[gen] = 0

            for sugg in suggestions:
                div_feats.append([diffs[sugg.generator]])

            max_val = max(max(d) for d in div_feats)
            if max_val != 0.0:
                for d in div_feats:
                    d[0] = d[0] / max_val

        return div_feats

    def avg_dist_evals(self, suggests, evals, psetting):
        """Average distance to all points so far evaluated.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        evals : list
            Already evaluated points.
        psetting : object
            Scenario parameters.

        Returns
        -------
        div_feats : list
            Average distances to all already evaluated points.
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

            max_val = max(max(d) for d in div_feats)
            if max_val != 0.0:
                for d in div_feats:
                    d[0] = d[0] / max_val

        else:
            div_feats = [[0] for _ in suggests]

        return div_feats

    def avg_dist_sel(self, suggests, psetting):
        """Average distance to points in the current selection.

        Parameters
        ----------
        suggestions : list
            Suggested points.
        psetting : object
            Scenario parameters.

        Returns
        -------
        div_feats : list
            Average distances to points in the current selection.
        """
        suggestions = copy.deepcopy(suggests)

        suggestions = hps.normalize_plus_cond_acc(suggestions, psetting)
        distances = hps.pairwise_distances(suggestions, suggestions)

        div_feats = []
        for dist in distances:
            div_feats.append([np.mean(dist)])

        max_val = max(max(d) for d in div_feats)
        if max_val != 0.0:
            for d in div_feats:
                d[0] = d[0] / max_val

        return div_feats

    def avg_dist_rel(self, suggests, evals, psetting, generators):
        """Average distances to relatives.

        Parameters
        ----------
        suggests : list
            Suggested points.
        evals : list
            Already evaluated points.
        psetting : object
            Scenario parameters.
        generators : list
            Available generators.

        Returns
        -------
        div_feats : list
            Computed features of suggested points.
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

            max_val = float(max(max(d) for d in div_feats))
            for d in div_feats:
                if max_val != 0.0:
                    d[0] = d[0] / max_val

        else:
            div_feats = [[0] for _ in suggests]

        return div_feats

    def expected_qual(self, suggs, sm, cot, surr, next_instance_set):
        """Expected quality of points.

        Parameters
        ----------
        suggests : list
            Suggested points.
        sm : object
            Surrogates.SurrogateManager().
        cot : int
            Cut off time (i.e. time limit).
        surr : str
            Which surrogate to use.
        next_instance_set : list
            Next instances that will be run.

        Returns
        -------
        dyn_feats : list
            Computed features of suggested points.
        """
        suggests = copy.deepcopy(suggs)
        dyn_feats = []
        try:
            expimp = sm.predict(surr, suggests, cot, next_instance_set)
            self.expimp = expimp

            for exim in expimp:
                for ei in exim.values():
                    dyn_feats.append([ei['qual']])

            max_val = float(max(max(d) for d in dyn_feats))
            for d in dyn_feats:
                if max_val != 0.0:
                    d[0] = d[0] / max_val

        except:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def prob_qual_improve(self, suggs, sm, cot, results, surr,
                          next_instance_set):
        """Probability of quality of points to improve.

        Parameters
        ----------
        suggests : list
            Suggested points.
        sm : object
            Surrogates.SurrogateManager().
        cot : int
            Cut off time (i.e. time limit).
        results : list
            Results of points evaluated so far.
        surr : str
            Which surrogate to use.

        Returns
        -------
        dyn_feats : list
            Computed features of suggested points.
        """
        suggests = copy.deepcopy(suggs)
        dyn_feats = []
        try:
            expimp = sm.pi(surr, suggests, cot, results, next_instance_set)

            for ei in expimp:
                dyn_feats.append(list(ei))

            max_val = float(max(max(d) for d in dyn_feats))
            for d in dyn_feats:
                if max_val != 0.0:
                    d[0] = d[0] / max_val

        except:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def uncertainty_improve(self, suggs, sm, cot, surr, next_instance_set):
        """Probability of quality of points to improve.

        Parameters
        ----------
        suggests : list
            Suggested points.
        sm : object
            Surrogates.SurrogateManager().
        cot : int
            Cut off time (i.e. time limit).
        surr : str
            Which surrogate to use.

        Returns
        -------
        dyn_feats : list
            Computed features of suggested points.
        """
        suggests = copy.deepcopy(suggs)
        dyn_feats = []
        try:
            expimp = self.expimp

            for exim in expimp:
                for ei in exim.values():
                    dyn_feats.append([ei['var']])

            max_val = float(max(max(d) for d in dyn_feats))
            for d in dyn_feats:
                if max_val != 0.0:
                    d[0] = d[0] / max_val

        except:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def expected_improve(self, suggs, sm, cot, surr, next_instance_set):
        """Probability of quality of points to improve.

        Parameters
        ----------
        suggests : list
            Suggested points.
        sm : object
            Surrogates.SurrogateManager().
        cot : int
            Cut off time (i.e. time limit).
        surr : str
            Which surrogate to use.

        Returns
        -------
        dyn_feats : list
            Computed features of suggested points.
        """
        suggests = copy.deepcopy(suggs)
        dyn_feats = []
        try:
            expimp = sm.ei(surr, suggests, next_instance_set)
            for ei in expimp:
                dyn_feats.append([ei])

            max_val = float(max(max(d) for d in dyn_feats))
            for d in dyn_feats:
                if max_val != 0.0:
                    d[0] = d[0] / max_val

        except:
            dyn_feats = [[0] for _ in suggests]

        return dyn_feats

    def surr_votes(self, dyn_feats):
        """Multiply surr features to get agreement features.

        Parameters
        ----------
        dyn_feats : list of np.ndarray
            Dynamic features.

        Returns
        -------
        dyn_feats : list of np.ndarray
            Extended dynamic features.
        """
        nr_surrs = len(Surrogates)
        new_feature_sets = []

        for k in range(0, len(dyn_feats[0]), nr_surrs):
            votes = [[] for _ in range(nr_surrs)]
            for i, _ in enumerate(dyn_feats):
                for j in range(nr_surrs):
                    votes[j].append([dyn_feats[i, k] * dyn_feats[i, k - 1]])
            new_feature_sets.append(votes)

        for nfs in new_feature_sets:
            for v in nfs:
                dyn_feats = np.concatenate((dyn_feats, v), axis=1)

        return dyn_feats

    def static_feature_gen(self, suggestions, epoch, max_epoch):
        """Generate static features.

        Parameters
        ----------
        suggestions : list
            Suggested configurations.
        epoch : int
            Current epoch.
        max_epoch : int
            Total number of epochs.

        Returns
        -------
        static_features : list
            Static features.
        """
        if self.logger is not None:
            static_time = time.time()

        static_feats = [[] for ii in range(len(suggestions))]

        # One-Hot encoded information of generator used for conf
        for s in range(len(suggestions)):
            for gt in range(len(self.Generator)):
                if suggestions[s].generator == self.Generator(gt + 1):
                    static_feats[s].append(1.0)
                else:
                    static_feats[s].append(0.0)

        # Ratio of current epoch and max. epochs
        for sf in range(len(static_feats)):
            static_feats[sf].append(epoch / max_epoch)

        if self.logger is not None:
            self.logger.info(f"Static features took {time.time() - static_time}\n\n")

        return np.array(static_feats)

    def dynamic_feature_gen(self, suggestions, data, predicted_quals, sm,
                            cot, results, next_instance_set):
        """Generate dynamic features.

        Parameters
        ----------
        suggestions : list
            Suggested configurations.
        data : object
            Contains historic data.
        predicted_quals : list of lists
            Predicted performance/quality for suggested configurations.
        sm : object
            Surrogates.SurrogateManager().
        cot : int
            Cut off time (i.e. time limit).
        results : list
            Results of points evaluated so far.

        Returns
        -------
        dyn_feats : list
            Dynamic features.
        """
        if self.logger is not None:
            all_dyn_time = time.time()
            dyn_one = time.time()

        # Features based on surrogates
        dyn_feats = self.expected_qual(suggestions, sm,
                                       cot, Surrogates.SMAC, next_instance_set)

        if self.logger is not None:
            self.logger.info(f"Dyn 1 features took {time.time() - dyn_one}\n\n")
            dyn_two = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.expected_qual(suggestions, sm,
                                               cot, Surrogates.GGApp,
                                               None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 2 features took {time.time() - dyn_two}\n\n")
            dyn_three = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.expected_qual(suggestions, sm,
                                               cot, Surrogates.CPPL,
                                               next_instance_set)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 3 features took {time.time() - dyn_three}\n\n")
            dyn_four = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.prob_qual_improve(suggestions, sm, cot,
                                                   results,
                                                   Surrogates.SMAC,
                                                   None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 4 features took {time.time() - dyn_four}\n\n")
            dyn_five = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.prob_qual_improve(suggestions, sm, cot,
                                                   results,
                                                   Surrogates.GGApp,
                                                   None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 5 features took {time.time() - dyn_five}\n\n")
            dyn_six = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.prob_qual_improve(suggestions, sm, cot,
                                                   results,
                                                   Surrogates.CPPL,
                                                   next_instance_set)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 6 features took {time.time() - dyn_six}\n\n")
            dyn_seven = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.uncertainty_improve(suggestions, sm, cot,
                                                     Surrogates.SMAC,
                                                     None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 7 features took {time.time() - dyn_seven}\n\n")
            dyn_eight = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.uncertainty_improve(suggestions, sm, cot,
                                                     Surrogates.GGApp,
                                                     None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 8 features took {time.time() - dyn_eight}\n\n")
            dyn_nine = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                            self.uncertainty_improve(suggestions, sm, cot,
                                                     Surrogates.CPPL,
                                                     next_instance_set)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 9 features took {time.time() - dyn_nine}\n\n")
            dyn_ten = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                           self.expected_improve(suggestions, sm, cot,
                                                 Surrogates.SMAC,
                                                 next_instance_set)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 10 features took {time.time() - dyn_ten}\n\n")
            dyn_eleven = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                           self.expected_improve(suggestions, sm, cot,
                                                 Surrogates.GGApp,
                                                 None)),
                           axis=1)

        if self.logger is not None:
            self.logger.info(f"Dyn 11 features took {time.time() - dyn_eleven}\n\n")
            dyn_twelve = time.time()

        dyn_feats = \
            np.concatenate((dyn_feats,
                           self.expected_improve(suggestions, sm, cot,
                                                 Surrogates.CPPL,
                                                 next_instance_set)),
                           axis=1)

        dyn_feats = self.surr_votes(dyn_feats)

        if self.logger is not None:
            self.logger.info(f"Dyn 12 features took {time.time() - dyn_twelve}\n\n")
            self.logger.info(f"All dyn features took {time.time() - all_dyn_time}\n\n")

        return np.array(dyn_feats)

    def diversity_feature_gen(self, suggestions, data, results, cot,
                              psetting, predicted_quals, evaluated):
        """Generate diversity features.

        Parameters
        ----------
        suggestions : list
            Suggested configurations.
        data : object
            Contains historic data.
        results : dict
            Qualities of configurations.
        cot : float
            Cut off time for tournaments.
        psetting : object
            Scenario parameters.
        predicted_quals : list
            Predicted qualities of points evaluated so far.
        evaluated : list
            All evaluated points so far.
        sm : object
            Initialized Surrogates.SurrogateManager().

        Returns
        -------
        div_feats : list
            Diversity features.
        """
        if self.logger is not None:
            div_time = time.time()

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

        div_feats = \
            np.concatenate((div_feats,
                           self.diff_pred_real_qual(suggestions, data,
                                                    predicted_quals,
                                                    results)),
                           axis=1)

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

        if self.logger is not None:
            self.logger.info(f"Div features took {time.time() - div_time}\n\n")

        return np.array(div_feats)
