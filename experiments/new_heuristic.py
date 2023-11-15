from collections import defaultdict
import numpy as np
import experiments.cython_test.aggregator
# from experiments.cython_test.aggregator import aggregator2 as get_start_time2
import pandas as pd
from functions.gantt import plot_schedule
import matplotlib.pyplot as plt
from functions.evaluation import drc_eval
from functions.verify import verify_solution
import os
import json
import time
import yaml


path_project = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
settings = yaml.load(open(os.path.join(path_project, "settings.yml"), encoding='UTF-8'), Loader=yaml.FullLoader)
paths = settings["paths"]

metric_to_variable_name = {"M-LAST": ("earliest_machine_operation_start", True),
                           "SPT": ("machine_processing", True),
                           "LPT": ("machine_processing", False),
                           "M-LQ": ("suitable_machine_queue", True),
                           "mLAST": ("earliest_machine_start", True),
                           "mLQ": ("machine_queue_size", True),
                           "wLAFT": ("earliest_worker_finish", True),
                           "wLQ": ("worker_queue_size", True),
                           "job": ("job", True),
                           "machine": ("machine", True),
                           "worker": ("worker", True)
                           }

strategy_to_metrics = {
    "M-LAST": {"operations": ["M-LAST", "SPT", "M-LQ"], "machines": ["mLAST", "mLQ"], "workers": ["wLAFT", "wLQ"]},
    "M-LQ": {"operations": ["M-LQ", "M-LAST", "LPT"], "machines": ["mLQ", "mLAST"], "workers": ["wLQ", "wLAFT"]},
    "LPT": {"operations": ["LPT", "M-LAST", "M-LQ"], "machines": ["mLAST", "mLQ"], "workers": ["wLAFT", "wLQ"]},
    "SPT": {"operations": ["SPT", "M-LAST", "M-LQ"], "machines": ["mLAST", "mLQ"], "workers": ["wLAFT", "wLQ"]}
}


def reduce_possible_insertions(insertions2, metrics, key=lambda test: (test["job"], test["operation"])):
    for metric in metrics:
        metric_to_compare, minimize_metric = metric_to_variable_name[metric]

        min_sign = 1 if minimize_metric else -1
        comparison_metric = 1e10*min_sign

        for possible_test in insertions2:
            test = insertions2[possible_test]
            if round(test[metric_to_compare], 4)*min_sign < round(comparison_metric, 4)*min_sign:
                comparison_metric = test[metric_to_compare]
                ties = [key(test)]
            elif round(test[metric_to_compare], 4) == round(comparison_metric, 4):
                ties.append(key(test))

        insertions2 = dict([(tie_key, insertions2[tie_key]) for tie_key in ties])
        assert(len(insertions2) > 0)

        if len(insertions2) == 1:
            break

    assert (len(insertions2) == 1)
    return insertions2


from functools import reduce
def get_start_time2(all_free_intervals, time_tracker_job):
    p = 0.0001
    limits = list(set([round(elem, 6) for elem in sum(sum(all_free_intervals, []), []) if elem >= time_tracker_job - p]))
    limits.sort()
    for start3 in limits:
        if reduce(lambda x, y: x + y, [1 if len([1 for st, et in slots if st-p <= start3 <= et+p]) else 0 for slots in all_free_intervals]) == 4:
            return start3

    for start3 in limits:
        if reduce(lambda x, y: x + y, [1 if len([1 for st, et in slots if st-p <= start3 <= et+p]) else 0 for slots in all_free_intervals]) == 4:
            return start3

    return time_tracker_job


def heuristic(strategy, instance_data):
    start_time = time.time()
    # 1 input : DRC FJSP instance
    # 2 output : schedule

    machine_metrics = strategy_to_metrics[strategy]["machines"] + ["machine"]
    operation_metrics = strategy_to_metrics[strategy]["operations"] + ["job"]
    worker_metrics = strategy_to_metrics[strategy]["workers"] + ["worker"]

    p = 6

    solution = {"order": [], 'machines': [], 'analysts': [], 'data': instance_data, "jop": []}

    all_operations = dict(sum([[((job["job"], test["test"]), test) for test in job["tests"]] for job in instance_data["jobs"]], []))
    tracker = {"jobs": defaultdict(int), "machines": defaultdict(int), "precedence": defaultdict(int)}
    key = {'start': 0, 'end': 1, 'job': 2, 'operation': 3, 'task': 4, 'resource': 5, 'resource_type': 6, 'queue_pos': 7}

    # Prepare initial empty schedule
    # columns = ['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type', 'queue_pos']
    sched = []
    for resource in ["machine", "analyst"]:
        for i in range(1, instance_data["n_{}s".format(resource)]+1):
            if len(sched):
                sched.append([0, 0, 0, 0, "0", i, resource, 0])
            else:
                sched = [[0, 0, 0, 0, "0", i, resource, 0]]

    # 3 while list of operations is not empty do
    while len(all_operations):
        # 4 for each operation do
        possible_insertions = {}
        already_scheduled = list(set([(s[key["job"]], s[key["operation"]]) for s in sched]))
        ops_todo = [[test["suitable_machines"] for test in job["tests"] if (job["job"], test["test"]) not in already_scheduled] for job in instance_data["jobs"]]

        for operation_key in all_operations:
            if tracker["precedence"][operation_key[0]] == int(operation_key[1]-1):  # Assure this operation can be processed
                job = operation_key[0]
                operation = operation_key[1]
                test = all_operations[operation_key]
                # 5 for each machine that can carry out the operation do
                possible_machines = {}
                for suitable_machine in test["suitable_machines"]:
                    # 6 compute earliest start time and finish time for carrying out the operation (taking into account already scheduled work)
                    earliest_start_time = max(tracker["machines"][suitable_machine], tracker["jobs"][job])
                    finish_time = earliest_start_time + test["machine_time"]

                    lq = len([s for s in sched if s[key["resource"]] == suitable_machine and s[key["resource_type"]] == "machine"])
                    mlq = sum([1/len(ops) for ops in sum(ops_todo, []) if suitable_machine in ops])

                    possible_machines[suitable_machine] = {
                        "earliest_machine_start": tracker["machines"][suitable_machine],
                        "earliest_machine_operation_start": earliest_start_time,
                        "earliest_machine_finish": finish_time,
                        "machine_queue_size": lq,
                        "suitable_machine_queue": mlq,
                        "job": job,
                        "operation": operation,
                        "machine": suitable_machine,
                        "machine_processing": test["machine_time"],
                    }

                # 7 attribute candidate machine to operation (with replacement)
                machine_to_insert = reduce_possible_insertions(possible_machines, machine_metrics, lambda x: x["machine"])
                possible_insertions[(job, operation)] = machine_to_insert[list(machine_to_insert.keys())[0]]

        # 8 choose operation based on operation selection criteria
        possible_insertions = reduce_possible_insertions(possible_insertions, operation_metrics)

        assert (len(possible_insertions) == 1)
        to_insert = list(possible_insertions.values())[0]
        to_insert_info = all_operations[(to_insert["job"], to_insert["operation"])]

        solution["order"].append(to_insert["job"])
        solution["machines"].append(to_insert["machine"])
        solution["jop"].append((to_insert["job"], to_insert["operation"]))

        # 9 for each worker that can carry out the operation do
        max_free = np.max([s[key["end"]] for s in sched]) + to_insert["machine_processing"]*2.0
        worker_insertions = {}

        machine_subset = [s for s in sched if s[key["resource"]] == to_insert["machine"] and s[key["resource_type"]] == "machine"]
        machine_subset.sort(key=lambda x: x[0])
        if len(machine_subset) == 1:
            machine_free_interval = [[machine_subset[0][key["end"]], max_free]]
        else:
            mach_time = to_insert["machine_processing"]
            machine_free_interval = [[0, machine_subset[0][0]-mach_time]] + [[machine_subset[i][1], round(machine_subset[i + 1][0]-mach_time, p)] for i in range(len(machine_subset) - 1)] + [[machine_subset[-1][1], max_free]]
            machine_free_interval = [[max(0, s), max(0, e)] for s, e in machine_free_interval if e >= s]

        for suitable_worker in to_insert_info["suitable_analysts"]:
            # 10 compute earliest start time and finish time for carrying out the operation (taking into account already scheduled work)

            # ['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type', 'queue_pos']
            subset = [s for s in sched if s[key["resource"]] == suitable_worker and s[key["resource_type"]] == "analyst"]
            subset.sort(key=lambda x: x[0])
            resource_queue_len = len(subset)
            if resource_queue_len == 1:
                resource_free_interval = [[subset[0][key["end"]], max_free]]
            else:
                subset = subset[1:]
                resource_free_interval = [[0, subset[0][0]]] + [[subset[i][1], subset[i+1][0]] for i in range(len(subset)-1)] + [[subset[-1][1], max_free]]
                resource_free_interval = [[max(0, s), max(0, e)] for s, e in resource_free_interval if max(0, e) > max(0, s)]

            tasks = [[ana_int["start"], ana_int["end"]] for ana_int in instance_data["jobs"][to_insert["job"]-1]["tests"][to_insert["operation"]-1]["analyst_times"]]
            task_len = [round(e - s, p) for s, e in tasks]

            all_free_intervals = [machine_free_interval] + \
                                 [[[round(S - tasks[t][0], p), round(E - task_len[t] - tasks[t][0], 2)] for S, E in resource_free_interval] for t in range(3)]
            all_free_intervals = [[[s, e] for s, e in interval if e >= s and e >= 0.0] for interval in all_free_intervals]

            all_limits = list(set(
                sum(sum(all_free_intervals, []), []) + [0] +
                [tracker["jobs"][job], tracker["machines"][to_insert["machine"]]] +
                [tracker["jobs"][to_insert["job"]]] + \
                [tracker["machines"][to_insert["machine"]]]
            ))
            all_limits.sort()

            new_intervals = []
            pre = 0.000001
            for interval in all_free_intervals:
                inter = []
                for intv in interval:
                    s_idx = np.argwhere([v+pre >= intv[0] for v in all_limits])[0][0]
                    e_idx = np.argwhere([v <= intv[1]+pre for v in all_limits])[-1][0]

                    new_ints = [[intv[0], all_limits[s_idx]]] + \
                               [[all_limits[i], all_limits[i+1]] for i in list(range(s_idx, e_idx))] + \
                               [[all_limits[e_idx], intv[1]]]
                    inter.append(new_ints)

                new_intervals.append(sum(inter, []))
            res_possible_start_time = get_start_time2(new_intervals, tracker["jobs"][to_insert["job"]])
            worker_insertions[suitable_worker] = {"earliest_worker_finish": res_possible_start_time+to_insert["machine_processing"],
                                                  "start_time": res_possible_start_time,
                                                  "worker_queue_size": resource_queue_len, "worker": suitable_worker, "tasks": tasks}

        #  11 choose worker based on machine selection criteria update schedule and remove chosen scheduled operation from list
        inserted_worker = reduce_possible_insertions(worker_insertions, worker_metrics, lambda x: x["worker"])
        inserted_worker_id = list(inserted_worker.keys())[0]
        operation_start_time = inserted_worker[inserted_worker_id]["start_time"]

        # Update trackers
        #['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type', 'queue_pos']
        tasks_str = ["setup", "intermediate", "teardown"]
        machine_starts = [round(val + operation_start_time, p) for val in [0] + [s for s, _ in worker_insertions[inserted_worker_id]["tasks"][1:]]]
        machine_ends = [round(val + operation_start_time, p) for val in
                        [s for s, _ in worker_insertions[inserted_worker_id]["tasks"][1:]] + [worker_insertions[inserted_worker_id]["tasks"][-1][1]]]
        for i in range(3):
            sched.append([machine_starts[i], machine_ends[i],
                          to_insert["job"], to_insert["operation"], tasks_str[i], to_insert["machine"], "machine", 0])

            sched.append([round(worker_insertions[inserted_worker_id]["tasks"][i][0]+operation_start_time, p),
                          round(worker_insertions[inserted_worker_id]["tasks"][i][1]+operation_start_time, p),
                          to_insert["job"], to_insert["operation"], tasks_str[i], inserted_worker[inserted_worker_id]["worker"], "analyst", 0])

        tracker["jobs"][to_insert["job"]] = inserted_worker[inserted_worker_id]["earliest_worker_finish"]
        tracker["machines"][to_insert["machine"]] = inserted_worker[inserted_worker_id]["earliest_worker_finish"]
        assert(to_insert["operation"]-1 == tracker["precedence"][to_insert["job"]])
        tracker["precedence"][to_insert["job"]] = to_insert["operation"]

        del all_operations[(to_insert["job"], to_insert["operation"])]

    final_sched = [row for row in sched if row[key["job"] != 0]]
    pd_sched = pd.DataFrame(final_sched, columns=['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type', 'queue_pos'])
    # assert(verify_solution(pd_sched, instance_data))
    result = drc_eval(pd_sched, instance_data)

    if settings["heuristic_settings"]["print_plot"]:
        plot_schedule(pd_sched)
        title = "{}: {}".format(strategy, result["total_completion_time"])
        plt.title(title)
        plt.show()

    pd_sched = pd_sched.replace("worker", "analyst")
    return time.time()-start_time, pd_sched, instance_data, result


if __name__ == "__main__":
    exp_parameters = settings['experiments']

    for n_jobs in exp_parameters["n_jobs"]:
        for n_machines in exp_parameters["n_machines"]:
            for n_workers in exp_parameters["n_workers"]:
                for n_sample_types in exp_parameters["n_sample_types"]:
                    for flexibility in exp_parameters["flexibility"]:
                        for i_replication in exp_parameters["replications"]:
                            instance = 'ex_j' + str(n_jobs) + '_' + \
                                       'm' + str(n_machines) + '_' + \
                                       'w' + str(n_workers) + '_' + \
                                       's' + str(n_sample_types) + '_' + \
                                       'f' + str(flexibility) + '_' + \
                                       'r' + str(i_replication)
                            filename = instance + '.json'
                            print('\nStart instance ', instance)

                            # Load instance data
                            with open(os.path.join(paths["instances"], filename)) as json_file:
                                instance_data = json.load(json_file)

                            for strategy in ["M-LAST", "M-LQ", "SPT", "LPT"]:
                                runtime, df, _, result = heuristic(strategy, instance_data)
                                print("{}: {}".format(strategy, result["total_completion_time"]))
                            break
