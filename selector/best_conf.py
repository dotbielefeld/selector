"""
This module sorts the configurations by performance within an iteration
and declares the overall best and saves it in a file.
"""

import json
import math


def read_run_hstory(path, timelimit):
    """
    This function reads in the run history.
    """
    
    # Opening JSON files
    f = open(f'{path}run_history.json')
    rh = json.load(f)
    f.close()

    with open(f'{path}trajectory.json', 'r') as handle:
        json_data = [json.loads(line) for line in handle]
    last_entry = list(json_data[-1].keys())[0]
    last_perf = 0
    for lp in list(rh[last_entry].values()):
        if math.isnan(lp):
            last_perf += timelimit
        else:
            last_perf += lp
    last_perf = last_perf / len(rh[last_entry])
    tr = {}
    for line in json_data:
        tr[list(line.keys())[0]] = list(line.values())[0]

    f.close()

    return rh, tr, last_entry, last_perf


def compute_performances(path, timelimit):
    """Compute and sort by performances."""

    performances = {}
    data, tr, last_entry, last_perf = read_run_hstory(path, timelimit)

    for k, v in data.items():
        len_v = len(v)
        if len_v not in performances:
            performances[len_v] = {}
        avg_perf = 0
        for perf in v.values():
            if math.isnan(perf):
                avg_perf += timelimit
            else:
                avg_perf += float(perf)
        performances[len_v][k] = avg_perf / len_v

    for perf in performances.keys():
        performances[perf] = \
            dict(sorted(performances[perf].items(), key=lambda x: x[1]))

    return dict(sorted(performances.items())), tr, data, last_entry, last_perf


def safe_best(path, timelimit):
    """Safe performances dict nd overall best config."""

    perfs, tr, data, best, last_perf = \
        compute_performances(path, timelimit)

    potential_best = list(perfs[list(perfs.keys())[-1]].keys())[0]

    while list(perfs.keys()) and potential_best != best:
        potential_best = list(perfs[list(perfs.keys())[-1]].keys())[0]
        if last_perf > perfs[list(perfs.keys())[-1]][potential_best]:
            if all(instance in data[best]
                    for instance in data[potential_best]) \
                    and potential_best in tr:
                potential_perf = []
                for instance in data[best]:
                    if math.isnan(data[potential_best][instance]):
                        potential_perf.append(timelimit)
                    else:
                        potential_perf.append(data[potential_best][instance])
                if sum(potential_perf) / len(potential_perf) < last_perf:
                    best = potential_best
                else:
                    del perfs[list(perfs.keys())[-1]][potential_best]
                    if len(perfs[list(perfs.keys())[-1]]) == 0:
                        del perfs[list(perfs.keys())[-1]]
            else:
                del perfs[list(perfs.keys())[-1]][potential_best]
                if len(perfs[list(perfs.keys())[-1]]) == 0:
                    del perfs[list(perfs.keys())[-1]]
        elif last_perf == perfs[list(perfs.keys())[-1]][potential_best]:
            break
        else:
            del perfs[list(perfs.keys())[-1]]

    overall_best = \
        {best: {'conf': tr[best], 'avg_perf': list(perfs.values())[-1][best]}}

    with open(f'{path}ranked_performances.json', 'w') as f:
        json.dump(perfs, f)

    with open(f'{path}overall_best.json', 'w') as f:
        json.dump(overall_best, f)

    return overall_best


if __name__ == "__main__":
    safe_best('./', 300)
