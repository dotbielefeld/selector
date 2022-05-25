"""This module contains point selection functions."""
import numpy as np
import copy
import scipy
import itertools
from selector.pool import ParamType


def transform_conf_vals(conf):
    """Transform configuration values from str to int.

    :param conf: list, configuration values
    :return conf: 1d array, transformed configuration values
    """
    for val in range(len(conf)):
        if type(conf[val]) == bool:
            if val:
                conf[val] = 1
            else:
                conf[val] = 0

    return np.array(conf)


def delete_conditionals(scenario, conf):
    """Delete conditional parameters, so that comparison is possible.

    Not all points have values for conditional params. In order to
    compute matching feature vectors, we omit conditional params.

    :param scenario: scenario object
    :param conf: nested list, configuration values
    :return conf: nested list, transformed configuration values
    """
    cond_params = list(scenario.conditionals.keys())
    for cond in cond_params:
        for c in range(len(conf)):
            if cond in conf[c].conf:
                del conf[c].conf[cond]

    return conf


def param_type_indices(scenario):
    """Get indices of categorical and other parameter types.

    :param scenario: scenario object
    :return cont_int_idc, cat_idc: arrays, indices of parameter types
    """
    cont_int_idc = []
    cat_idc = []
    params = scenario.parameter

    idx = 0

    for p in range(len(params)):
        # Record index if param is cat and not in conditional
        if params[p].type == ParamType.categorical and \
                not params[p].condition:
            cat_idc.append(idx)
        # Record index if param is cont/int and not in conditional
        elif not params[p].condition:
            cont_int_idc.append(idx)
        # If param is in conditionals, do not record index, reset index
        # to make sure indices are adjusted to shorter lists, because
        # params which are in conditionals are deleted by delete_conditionals()
        else:
            idx -= 1
        idx += 1

    return np.array(cont_int_idc), np.array(cat_idc)


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

    return np.array(relatives)


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
    suggested = delete_conditionals(scenario, suggested)

    selected_points = []
    smselected_points = []

    sugg_points = []
    for sp in suggested:
        conf = transform_conf_vals(list(sp.conf.values()))
        sugg_points.append(conf)

    sugg_points = np.array(sugg_points)
    cont_int_idc, cat_idc = param_type_indices(scenario)

    # Compute euclidean distances with continous and integer values
    eucd = scipy.spatial.distance.cdist(sugg_points[:, cont_int_idc],
                                        sugg_points[:, cont_int_idc],
                                        metric='euclidean')

    # Compute Hamming distance with categorical values, if present
    if not cat_idc.size == 0:
        hamd = []
        for idxf, frompoint in enumerate(sugg_points[:, cat_idc]):
            hd = []
            for idxt, topoint in enumerate(sugg_points[:, cat_idc]):
                if idxf != idxt:
                    hd.append(scipy.spatial.distance.hamming(frompoint,
                                                             topoint,
                                                             w=None))
            hamd.append(hd)
        hamd = np.array(hamd)
    else:
        hamd = np.array([0 for _ in sugg_points])

    distances = np.hstack((eucd, hamd))

    # Run simulation for every point requested
    for psel in range(npoints):

        sfreq = simulation(suggested, features, max_evals, smselected_points,
                           weights, npoints, distances, relatives)
        sidx = np.argmax(sfreq)
        selected_points.append(suggested_intact[sidx])
        smselected_points.append(sidx)
        del suggested[sidx]
        weights = np.delete(weights, sidx, axis=0)
        features = np.delete(features, sidx, axis=0)
        distances = np.delete(distances, sidx, axis=0)

    return selected_points
