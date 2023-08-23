"""This module contains point selection functions."""
import numpy as np
import copy
import itertools
from selector.pool import ParamType


def get_relatives(suggested):
    """Get information of relations of suggested points by generator tag.

    :param suggested: list of suggested points
    :return relatives: nested array, indices of related points
    """
    relatives = []
    for s in suggested:
        gen_type = s.generator
        index_list = []
        index_list = [idx for idx, sugg in enumerate(suggested)
                      if sugg != s and sugg.generator == gen_type]
        relatives.append(index_list)

    return np.array(relatives, dtype=object)


def distance_stats(smfeatures, distances):
    """Compute distances statistics.

    :param suggested: list, list of suggested points
    :param distances: list, distance values
    :return smfeatures: array, new features for simulation
    """
    smflen = len(smfeatures[0])
    smfeatures = np.hstack((smfeatures, np.mean(distances, axis=1).reshape(
                            len(distances), 1)))
    smfeatures = np.hstack((smfeatures, np.mean(distances * distances,
                            axis=1).reshape(len(distances), 1)))
    smfeatures = np.hstack((smfeatures, np.std(distances, axis=1).reshape(
                            len(distances), 1)))
    mindist = np.min(distances, axis=1)
    smfeatures = np.hstack((smfeatures, (smfeatures[:, smflen] -
                            mindist).reshape(len(distances), 1)))

    return smfeatures


def simulation(suggested, features, max_evals, selected_points, weights,
               npoints, distances, relatives):
    """Run simulations of config selection.

    :param suggested: list, list of configs/points to select from
    :param features: nested list, features of configs/points
    :param max_eval: int, number of simulation runs per selected point
    :return sfreq: list, how often configs/points were selected in sim
    """
    sugg = list(range(len(suggested)))
    sfreq = np.zeros(len(sugg))

    for evaluation in range(max_evals):
        smsel = copy.copy(selected_points)
        smsugg = copy.copy(sugg)
        smfeatures = copy.copy(features)
        smweights = copy.copy(weights)
        smdistances = copy.copy(distances)

        for selpoint in range(len(selected_points), npoints):

            # After the first point is chosen
            if selpoint > 0:
                # Diversity features to selected points
                simseldist = smdistances[:, smsel]
                smfeatures = distance_stats(smfeatures, simseldist)

            rel_sel = list(itertools.chain.from_iterable(relatives[sel]
                                                         for sel in smsel))
            if rel_sel:
                # Diversity features to selected and related points
                simrelseldist = smdistances[:, rel_sel]
                smfeatures = distance_stats(smfeatures, simrelseldist)

            # Min-max normalization
            minf = np.min(smfeatures, axis=0)
            maxf = np.max(smfeatures, axis=0)
            diff = maxf - minf
            eq = np.where(minf == maxf)[0]
            ge = np.setdiff1d(np.arange(smfeatures.shape[1]), eq,
                              assume_unique=True)
            smfeatures[:, ge] = (smfeatures[:, ge] - minf[ge]) / diff[ge]
            # set no variance features to 0, except for the first
            smfeatures[:, eq[1:]] = 0

            # Probability distribution based on scores
            s_w = 1.0 / (1.0 + np.exp(np.sum(smfeatures *
                         smweights[:, 0:len(smfeatures[0])], axis=1)))

            # Scores based on probability distribution
            scores = np.maximum(0, np.minimum(1, s_w))

            # Select with probability according to scores
            if np.sum(scores) > 0:
                selprob = scores / np.sum(scores)
                selected = np.random.choice(smsugg, 1, p=selprob.tolist())[0]
                selected_idx = smsugg.index(selected)
            else:
                selected = np.random.choice(smsugg, 1)[0]
                selected_idx = smsugg.index(selected)

            # Update frequency of selections
            sfreq[selected] += 1

            # Update point selection within simulation run
            smsel.append(selected_idx)

            # Make sure selected points cannot be selected again in simulation
            del smsugg[selected_idx]
            smfeatures = np.delete(smfeatures, selected_idx, axis=0)
            smweights = np.delete(smweights, selected_idx, axis=0)
            smdistances = np.delete(smdistances, selected_idx, axis=0)
            smfeatures = copy.copy(features[0:len(smfeatures)])

    return sfreq


def normalize_plus_cond_acc(sugg, s):
    """Normalize and account for conditionals.

    :param sugg: list, configuration values
    :param s: scenario
    :return sugg: suggested configuration with normalized and adjusted values
    """
    maximums = {}
    cat_params = []
    if isinstance(s, list):
        psetting = s
    else:
        psetting = s.parameter

    for param in psetting:
        if param.type == ParamType.categorical:
            if len(param.bound) > 2:
                if isinstance(param.bound[0], (str, np.str_)):
                    maximums[param.name] = len(param.bound)
                else:
                    maximums[param.name] = \
                        float(param.bound[len(param.bound) - 1])
            else:
                maximums[param.name] = 1
            cat_params.append(param.name)
        else:
            maximums[param.name] = param.bound[len(param.bound) - 1]

    for point in sugg:
        for key, _ in point.conf.items():
            if key in cat_params:
                if point.conf[key] is True:
                    point.conf[key] = 1
                else:
                    point.conf[key] = 0

    for key, val in maximums.items():
        for point in sugg:
            if key in point.conf:
                if point.conf[key] is None:
                    pass
            elif key in point.conf and maximums[key] > 0:
                point.conf[key] = point.conf[key] / maximums[key]
            elif key in point.conf and maximums[key] < 0:
                point.conf[key] = maximums[key] / point.conf[key]
            elif key not in point.conf:
                point.conf[key] = None

    return sugg


def pairwise_distances(sugg_i, sugg_j):
    """Compute pairwise distances.

    :param sugg_i: list, configuration values
    :param sugg_j: list, configuration values
    :return m: pairwise distances
    """
    m = np.zeros((len(sugg_i), len(sugg_j)))
    for i, s_i in enumerate(sugg_i):
        for j, s_j in enumerate(sugg_j):
            s = 0
            for key in s_i.conf:
                if (s_i.conf[key] is None and s_j.conf[key]
                        is not None) or \
                        (s_i.conf[key] is not None and s_j.conf[key]
                            is None):
                    s = s + 1
                elif (s_i.conf[key] is None and s_j.conf[key]
                        is None):
                    s = s + 0
                else:
                    if isinstance(s_i.conf[key], str) or \
                            isinstance(s_j.conf[key], str):
                        s = s + (float(s_i.conf[key]) - float(s_j.conf[key]))**2
                    else:
                        s = s + (s_i.conf[key] - s_j.conf[key])**2

            m[i, j] = s**0.5

    return np.array(m)


def select_point(scenario, suggested, max_evals, npoints, pool, epoch,
                 max_epoch, features, weights, seed):
    """Generate features and run simultion.

    :param suggested: list, list of configs/points to select from
    :param max_eval: int, number of simulation runs per selected point
    :param npoints: int, number of configs/points requested
    :param pool: list, list of configs/pints to select from
    :param epoch: int, current epoch
    :param max_epoch: int, number of total epochs
    :return selected_points: list, ids of selected configs/points
    """
    if seed:
        np.random.seed(seed)

    relatives = get_relatives(suggested)

    suggested_intact = copy.copy(suggested)

    # Not all points have values for conditional params. In order to
    # compute matching feature vectors, we omit conditional params.
    sugg = copy.deepcopy(suggested)
    sugg = normalize_plus_cond_acc(sugg, scenario)

    distances = pairwise_distances(sugg, sugg)

    selected_points = []
    smselected_points = []

    # Run simulation for every point requested
    for psel in range(npoints):

        sfreq = simulation(suggested, features, max_evals, smselected_points,
                           weights, npoints, distances, relatives)
        sidx = np.argmax(sfreq)
        selected_points.append(suggested_intact[sidx])
        del suggested_intact[sidx]
        smselected_points.append(sidx)
        del suggested[sidx]
        weights = np.delete(weights, sidx, axis=0)
        features = np.delete(features, sidx, axis=0)
        distances = np.delete(distances, sidx, axis=0)

    return selected_points
