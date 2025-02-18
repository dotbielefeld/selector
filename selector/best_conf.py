"""
This module sorts the configurations by performance within an iteration
and declares the overall best and saves it in a file.
"""

import json
import math


def read_run_hstory(path, penalty):
    """
    This function reads in the run history.

    : param path: path to log folder
    : param penalty: penalty for non-solved instance
    return: runhistory, tournament history
    """
    
    # Opening JSON files
    f = open(f'{path}run_history.json')
    rh = json.load(f)
    f.close()

    f = open(f'{path}tournament_history.json')
    json_data = json.load(f)
    f.close()

    th = {}
    for tournament_id, tournament in json_data.items():
        for conf in tournament['best_finisher']:
            th.update({conf['id']: conf['conf']})
        for conf in tournament['worst_finisher']:
            th.update({conf['id']: conf['conf']})

    return rh, th


def compute_performances(path, penalty):
    """
    Compute and sort by performances.

    : param path: path to log folder
    : param penalty: penalty for non-solved instance
    return: performances of winners, tournament history, runhistory
    """

    performances = {}
    data, th = read_run_hstory(path, penalty)

    for k, v in data.items():
        len_v = len(v)
        if len_v not in performances:
            performances[len_v] = {}
        avg_perf = 0
        for perf in v.values():
            if math.isnan(perf):
                avg_perf += penalty
            else:
                avg_perf += float(perf)
        performances[len_v][k] = avg_perf / len_v

    for perf in performances.keys():
        for conf in performances[perf]:
            pen_perf = {k: (penalty 
                        if math.isnan(v) else v)
                        for k, v in data[conf].items()}
            performances[perf][conf] = sum(pen_perf.values()) / len(pen_perf)
        
        performances[perf] = \
            dict(sorted(performances[perf].items(), key=lambda x: x[1]))

    return dict(sorted(performances.items())), th, data


def safe_best(path, penalty):
    """
    Safe performances dict nd overall best config.

    : param path: path to log folder
    : param penalty: penalty for non-solved instance
    return: overall best configuration
    """

    perfs, th, data = \
        compute_performances(path, penalty)

    inst_sizes = list(perfs.keys())
    inst_sizes.reverse()
    for i, isize in enumerate(inst_sizes):
        if len(perfs[isize]) > 1:
            nn_1 = sum(1 for v in data[list(perfs[isize].keys())[0]].values()
                       if not math.isnan(v))
            nn_sum_1 = sum(v for v in data[list(perfs[isize]
                           .keys())[0]].values() 
                           if not math.isnan(v))
            nn_2 = sum(1 for v in data[list(perfs[inst_sizes[i + 1]]
                       .keys())[0]].values() if not math.isnan(v))
            nn_sum_2 = sum(v for v in data[list(perfs[inst_sizes[i + 1]]
                           .keys())[0]].values()
                           if not math.isnan(v))

            if (perfs[isize][list(perfs[isize].keys())[0]] 
                    < perfs[inst_sizes[i + 1]]
                    [list(perfs[inst_sizes[i + 1]].keys())[0]]):
                best = list(perfs[isize].keys())[0]
                best_perf = perfs[isize][best]
                break

            elif nn_1 < nn_2 and nn_sum_1 < nn_sum_2:
                best = list(perfs[isize].keys())[0]
                best_perf = perfs[isize][best]

            elif nn_1 > nn_2 and nn_sum_1 > nn_sum_2:
                best = list(perfs[inst_sizes[i + 1]].keys())[0]
                best_perf = perfs[inst_sizes[i + 1]][best]

            else:
                continue
        else:
            continue

    overall_best = \
        {best: {'conf': th[best], 'avg_perf': best_perf}}

    with open(f'{path}ranked_performances.json', 'w') as f:
        json.dump(perfs, f)

    with open(f'{path}overall_best.json', 'w') as f:
        json.dump(overall_best, f)

    return overall_best


if __name__ == "__main__":
    pass
