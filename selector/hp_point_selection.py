"""This module contains point selection functions."""
import numpy as np
import copy
import scipy
import itertools
from selector.pool import ParamType
from selector.selection_features import FeatureGenerator


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
        if params[p].type == ParamType.categorical and \
                not params[p].condition:
            cat_idc.append(idx)
        elif not params[p].condition:
            cont_int_idc.append(idx)
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
        for i in range(len(suggested)):
            if suggested[i] != s and suggested[i].generator == gen_type:
                index_list.append(i)
        relatives.append(index_list)

    return np.array(relatives)


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
                smflen = len(smfeatures[0])
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.mean(simseldist, axis=1), axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.mean(simseldist * simseldist,
                                       axis=1), axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.std(simseldist, axis=1), axis=1)
                mindist = np.min(simseldist, axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       smfeatures[:, smflen] - mindist, axis=1)

            rel_sel = list(itertools.chain.from_iterable(relatives[sel]
                                                         for sel in smsel))
            if rel_sel:
                # Diversity features to selected and related points
                simrelseldist = smdistances[:, rel_sel]
                smflen = len(smfeatures[0])
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.mean(simrelseldist, axis=1), axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.mean(simrelseldist * simrelseldist,
                                       axis=1), axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       np.std(simrelseldist, axis=1), axis=1)
                smfeatures = np.insert(smfeatures, len(smfeatures[0]),
                                       smfeatures[:, smflen] -
                                       np.min(simrelseldist, axis=1), axis=1)

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

            # Probability distribution according to features
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
                 max_epoch, weights, seed):
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

    suggested = delete_conditionals(scenario, suggested)

    fg = FeatureGenerator()
    features = fg.static_feature_gen(pool, epoch, max_epoch)
    selected_points = []
    smselected_points = []

    sugg_points = []
    for sp in suggested:
        conf = transform_conf_vals(list(sp.conf.values()))
        sugg_points.append(conf)

    sugg_points = np.array(sugg_points)
    cont_int_idc, cat_idc = param_type_indices(scenario)

    # Compute euclidean distances wit continous and integer values
    eucd = scipy.spatial.distance.cdist(sugg_points[:, cont_int_idc],
                                        sugg_points[:, cont_int_idc],
                                        metric='euclidean')

    # Compute Hamming distance with categorical values, if present
    if cat_idc:
        hamd = scipy.spatial.distance.hamming(sugg_points[:, cat_idc],
                                              sugg_points[:, cat_idc],
                                              w=None)
    else:
        hamd = 0

    distances = eucd + hamd

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
