import os
import yaml
import json
import pandas as pd
import numpy as np
import datetime
import random
import copy
import matplotlib.pyplot as plt

from functions.gantt import plot_schedule
from experiments.new_heuristic import heuristic
from functions.evaluation import drc_eval
from collections import defaultdict, deque
from functions.read_results import read_past_results, all_results_done, get_latest_result, save_result
from experiments.cython_test.aggregator import aggregator2 as get_start_time2
from functions.verify import verify_solution


path_project = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
settings = yaml.load(open(os.path.join(path_project, "settings.yml"), encoding='UTF-8'), Loader=yaml.FullLoader)
paths = settings["paths"]
param = settings["parameters"]


def encode(df, this_instance_data):
    df2 = df.loc[df.resource_type == "machine"].sort_values(['start', 'resource'], ascending=[True, True])
    ov = df2.job.values
    df3 = df.loc[df.resource_type == "machine"].sort_values(['job', 'operation'], ascending=[True, True])
    mv = df3.resource.values
    order = [(row.job, row.operation, row.task, "analyst") for index, row in df3.iterrows()]
    df4 = df.set_index(["job", "operation", "task", "resource_type"])
    av = [df4.loc[oidx].resource for oidx in order]
    encoded_solution = {"order": ov, 'machines': mv, 'analysts': av, 'data': this_instance_data, "jop": order}
    return encoded_solution


def get_resource_idx(job, operation, task, s):
    # Return machine or analyst vector index for given job, operation and task
    return np.where([j == job and op == operation and task == t for j, op, t, a in s["jop"]])[0][0]


def get_operation_idx(job, operation, task, s):
    # Return position in order vector
    task_dict = {"setup": 1, "intermediate": 2, "teardown": 3}
    freq = (operation-1)*3 + task_dict[task]-1
    idx = np.where([elem == job for elem in s["order"]])[0][freq]
    return idx


def decode(solution):
    s = copy.deepcopy(solution)
    columns = ['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type', 'queue_pos']
    data = []
    operation_tracker = defaultdict(int)
    time_tracker = {"machines": defaultdict(int), "analysts": defaultdict(int), "job": defaultdict(int)}

    # Force machine tasks of same operation in sequence
    resource_type = "machines"
    for resource in set(solution[resource_type]):
        indexes = [order for order, tresource in zip(solution["jop"], solution[resource_type]) if tresource == resource]

        full_combos = [(job, op) for job, op, a, b in indexes]

        combos = []
        [combos.append((job, op)) for job, op, a, b in indexes if (not (job, op) in combos)]

        for combo in combos:
            idxs = np.where([each_combo == combo for each_combo in full_combos])[0]
            assert((idxs[0]+2) == (idxs[1]+1) == idxs[2])

    # Check machine resource same for all operations
    sorder = list(s["order"])
    while len(sorder) > 0:
        job = sorder.pop(0)

        # Remove next two operations
        sorder.remove(job)
        sorder.remove(job)

        operation_idx = operation_tracker[job]
        operation_tracker[job] += 1
        operation = operation_tracker[job]

        resource_idx_1 = get_resource_idx(job, operation, "setup", s)
        resource_idx_2 = get_resource_idx(job, operation, "intermediate", s)
        resource_idx_3 = get_resource_idx(job, operation, "teardown", s)

        machine = solution["machines"][resource_idx_1]
        analyst_1 = solution["analysts"][resource_idx_1]
        analyst_2 = solution["analysts"][resource_idx_2]
        analyst_3 = solution["analysts"][resource_idx_3]

        operation_info = solution["data"]["jobs"][job - 1]["tests"][operation_idx]["analyst_times"]
        total_machine_processing = solution["data"]["jobs"][job-1]["tests"][operation_idx]["machine_time"]
        machine_processing_1 = round(operation_info[1]["start"], 4)
        machine_processing_2 = round(operation_info[2]["start"] - operation_info[1]["start"], 4)
        machine_processing_3 = round(total_machine_processing - operation_info[2]["start"], 4)
        analyst_offset_1 = operation_info[0]["start"]
        analyst_offset_2 = operation_info[1]["start"]
        analyst_offset_3 = operation_info[2]["start"]
        analyst_processing_1 = round(operation_info[0]["end"] - operation_info[0]["start"], 4)
        analyst_processing_2 = round(operation_info[1]["end"] - operation_info[1]["start"], 4)
        analyst_processing_3 = round(operation_info[2]["end"] - operation_info[2]["start"], 4)

        # Get leftmost possible start time
        # if False:  # Simple leftmost
        #     start = np.max([time_tracker["machines"][machine],
        #                     time_tracker["analysts"][analyst_1],
        #                     time_tracker["analysts"][analyst_2],
        #                     time_tracker["analysts"][analyst_3],
        #                     time_tracker["job"][job]])

        max_overall = np.max([row[1] for row in data]) if len(data) > 0 else 0
        start = -1
        # Get all free time slots for each resource
        all_free_intervals = []
        p = 0.0001

        for res_type, res_id, ptime, offset in [
            # res type, res id,     proc.time,          offset
            ["machine", machine, total_machine_processing, 0],
            ["analyst", analyst_1, analyst_processing_1, analyst_offset_1],
            ["analyst", analyst_2, analyst_processing_2, analyst_offset_2],
            ["analyst", analyst_3, analyst_processing_3, analyst_offset_3]]:

            # Define possible free intervals (offset to always compare start time!)
            subset = [row for row in data if (row[6] == res_type) and (row[5] == res_id)]
            subset.sort()
            if len(subset):
                free_intervals = [[0-offset, subset[0][0]-offset-ptime]] if (0-offset - p < subset[0][0]-offset-ptime) else []
                free_intervals += [[subset[i][1]-offset, subset[i+1][0]-offset-ptime] for i in range(len(subset)-1) if (subset[i][1] - p < subset[i+1][0]) and ((subset[i+1][0] - subset[i][1]) > .05)]
                free_intervals += [[subset[-1][1]-offset, max_overall+0.1]]

                # Remove negatives
                free_intervals = [l for l in free_intervals if l[1] >= (0-p)]
            else:
                free_intervals = [[0, max_overall+.1]]

            all_free_intervals.append(free_intervals)

        start = get_start_time2(all_free_intervals, time_tracker["job"][job])
        analyst_start_1 = round(start + operation_info[0]["start"], 4)
        analyst_start_2 = round(start + operation_info[1]["start"], 4)
        analyst_start_3 = round(start + operation_info[2]["start"], 4)
        analyst_end_1 = round(analyst_start_1 + analyst_processing_1, 4)
        analyst_end_2 = round(analyst_start_2 + analyst_processing_2, 4)
        analyst_end_3 = round(analyst_start_3 + analyst_processing_3, 4)

        time_tracker["analysts"][analyst_1] = max(time_tracker["analysts"][analyst_1], analyst_end_1)
        time_tracker["analysts"][analyst_2] = max(time_tracker["analysts"][analyst_2], analyst_end_2)
        time_tracker["analysts"][analyst_3] = max(time_tracker["analysts"][analyst_3], analyst_end_3)

        machine_start_2 = round(start + machine_processing_1, 4)
        machine_start_3 = round(machine_start_2 + machine_processing_2, 4)

        machine_end_1 = round(start + machine_processing_1, 4)
        machine_end_2 = round(machine_start_2 + machine_processing_2, 4)
        machine_end_3 = round(machine_start_3 + machine_processing_3, 4)
        machine_end = round(start + machine_processing_1 + machine_processing_2 + machine_processing_3, 4)

        time_tracker["machines"][machine] = machine_end
        time_tracker["job"][job] = machine_end

        data.append([start, machine_end_1, job, operation, "setup", machine, "machine", 0])
        data.append([analyst_start_1, analyst_end_1, job, operation, "setup", analyst_1, "analyst", 0])

        data.append([machine_start_2, machine_end_2, job, operation, "intermediate", machine, "machine", 0])
        data.append([analyst_start_2, analyst_end_2, job, operation, "intermediate", analyst_2, "analyst", 0])

        data.append([machine_start_3, machine_end_3, job, operation, "teardown", machine, "machine", 0])
        data.append([analyst_start_3, analyst_end_3, job, operation, "teardown", analyst_3, "analyst", 0])

    df = pd.DataFrame(data=data, columns=columns).astype({"job": "int", "operation":"int", "resource":"int"})
    return df


def tabu_entry(solution):
    return np.concatenate((solution["order"], solution["machines"], solution["analysts"]))


def check_swap_feasibility(solution, swap):
    # Assure balanced solution
    flat_list = {}
    for resource in ['analysts', 'machines']:
        flat_list[resource] = sum([solution[resource][key] for key in solution[resource].keys()], [])
    assert (len(flat_list['machines']) == len(flat_list['analysts']) and len(set(flat_list['machines'])) == len(set(flat_list['analysts'])))


def fitness(solution):
    value_out = drc_eval(decode(solution))['total_completion_time']
    return value_out


def tabu_search(df_schedule, instance_data, instance,
                max_steps=param["tabu_search"]["max_steps"],
                ls_steps=param["tabu_search"]["ls_steps"],
                tabu_list_size=param["tabu_search"]["tabu_list_size"]):
    start_time = datetime.datetime.now()
    random.seed(datetime.datetime.now().second)
    print("max steps: ", max_steps, ", time: ", start_time)
    # Encode
    if any(df_schedule['resource_type'] == "worker"):
        df_schedule = df_schedule.replace(to_replace="worker", value="analyst")
    initial_solution = encode(df_schedule, instance_data)

    verify_solution(df_schedule, instance_data)
    best_value = drc_eval(df_schedule)['total_completion_time']
    print('Initial c: ', best_value)
    initial_best_value = best_value
    print_tabu = False

    if print_tabu:
        sched_fig, sched_ax = plot_schedule(df_schedule, legend="heuristic")
        sched_ax.set_title('Swap solution, c=' + str(drc_eval(df_schedule)))
        sched_fig.show()

    if print_tabu:
        sched_fig, sched_ax = plot_schedule(df_schedule, legend="heuristic")
        sched_ax.set_title(instance + '\nInitial solution, c=' + str(drc_eval(df_schedule)))
        sched_fig.show()

    s_best = copy.deepcopy(initial_solution)
    s_best_fitness = fitness(s_best)
    best_candidate = copy.deepcopy(initial_solution)
    tabu_list = deque([tabu_entry(initial_solution)], maxlen=tabu_list_size)

    generated_list = []
    visited_list = []

    plot_data_best = [s_best_fitness]
    plot_data_avg = [s_best_fitness]
    plot_data_worst = [s_best_fitness]
    visited_list.append(best_candidate)

    no_improvement_counter = 0

    for i in range(max_steps):
        if (i % round(max_steps/5.0,0)) == 0:
            print("Tabu Search iteration ", i)
        s_neighborhood = []
        for _ in range(ls_steps):
            # Create LS solution
            new_solution = new_swap(best_candidate)
            s_neighborhood.append(copy.deepcopy(new_solution))

        generated_list.append(s_neighborhood)  # Add to total generated list

        best_candidate = copy.deepcopy(s_neighborhood[0])
        best_candidate_fitness = fitness(best_candidate)
        worst_candidate_fitness = best_candidate_fitness
        candidates_fitness = []
        for s_candidate in s_neighborhood:
            s_candidate_fitness = fitness(s_candidate)
            candidates_fitness.append(s_candidate_fitness)
            if s_candidate_fitness < best_candidate_fitness:
                if not np.any([np.all([t == e for t, e in zip(tabu_entry(s_candidate), elem)]) for elem in tabu_list]):
                    best_candidate = copy.deepcopy(s_candidate)
                    best_candidate_fitness = s_candidate_fitness

            if s_candidate_fitness > worst_candidate_fitness:
                worst_candidate_fitness = s_candidate_fitness
        if best_candidate_fitness < s_best_fitness:
            s_best = copy.deepcopy(best_candidate)
            s_best_fitness = best_candidate_fitness
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        tabu_list.append(tabu_entry(best_candidate))

        plot_data_best.append(s_best_fitness)
        plot_data_avg.append(np.mean(candidates_fitness))
        plot_data_worst.append(worst_candidate_fitness)
        visited_list.append(best_candidate)

        if settings["parameters"]["tabu_search"]["stop_early"] == 1 and i > max_steps*.1 and no_improvement_counter > .66*i:
            print("Tabu search converged at iteration ", i)
            break

        if (datetime.datetime.now()-start_time).seconds > 3600:
            print("Tabu Search exceeded 1h, stopping at iteration ", i)
            break

    if print_tabu:
        plt.plot(plot_data_best)
        plt.plot(plot_data_avg)
        plt.plot(plot_data_worst)
        plt.legend(["Best", "Average", "Worst"])

    # Decode
    decoded_solution = decode(s_best)
    if initial_best_value <= s_best_fitness:
        print('TABU gave no improvement to ', initial_best_value)
        final_solution = df_schedule
    else:
        print('TABU improved from ', initial_best_value, 'to ', s_best_fitness)
        final_solution = decoded_solution

    end_time = datetime.datetime.now()
    duration_time = (end_time - start_time).total_seconds()

    unique_generated_solutions = len(set([str(value['machines']) + str(value['analysts']) for value in sum(generated_list, [])]))
    unique_visited_solutions = len(set([str(value['machines']) + str(value['analysts']) for value in visited_list]))

    if print_tabu:
        plt.title("Tabu Search: steps {}, LS {}, list {}, gen {}, vis {}".format(
            max_steps, ls_steps, tabu_list_size, unique_generated_solutions, unique_visited_solutions))
        plt.show()
    verify_solution(final_solution, instance_data)

    parameter_dict = {"max_steps": max_steps, "ls_steps": ls_steps, "tabu_list_size": tabu_list_size}
    return final_solution, {"parameters": parameter_dict,
                            "time": duration_time,
                            "unique_generated_solutions": unique_generated_solutions,
                            "unique_visited_solutions": unique_visited_solutions,
                            "initial_solution": initial_solution,
                            "total_steps_no_imp": no_improvement_counter,
                            "total_steps": i}


def fix_order(s, resources):
    for new_resource in resources:
        # Readjust order vector
        unordered_jop = [jop for jop, machine in zip(s["jop"], s["machines"]) if machine == new_resource]
        unordered_order_idx = [(ujop[0], ujop[1], get_operation_idx(ujop[0], ujop[1], ujop[2], s)) for ujop in unordered_jop]
        ordered_order_idx = sorted(unordered_order_idx, key=lambda x: x[2])
        original_order = [elem[0] for elem in ordered_order_idx]
        original_idx = [elem[2] for elem in ordered_order_idx]

        final = []
        while len(original_order):
            jinsert = original_order[0]
            original_order.remove(jinsert)
            original_order.remove(jinsert)
            original_order.remove(jinsert)
            final.append(jinsert)
            final.append(jinsert)
            final.append(jinsert)

        for i in range(len(final)):
            s["order"][original_idx[i]] = final[i]
    return s


def new_swap(solution):
    s = copy.deepcopy(solution)
    resource_idx = random.randrange(len(solution["order"]))
    job_op = [[(jobs["job"], op["test"]) for op in jobs["tests"]] for jobs in s["data"]["jobs"]]
    rand_number = random.random()
    if rand_number < .6666:
        if rand_number < .3333:
            resource = "machines"
        else:
            resource = "analysts"

        job, op, _, _ = solution["jop"][resource_idx]

        possible_resources = s["data"]["jobs"][job-1]["tests"][op-1]["suitable_"+resource].copy()
        if len(possible_resources) > 1:
            possible_resources.remove(solution[resource][resource_idx])
            new_resource = random.sample(set(possible_resources), 1)[0]

            if resource == "machines":
                s[resource][get_resource_idx(job, op, "setup", solution)] = new_resource
                s[resource][get_resource_idx(job, op, "intermediate", solution)] = new_resource
                s[resource][get_resource_idx(job, op, "teardown", solution)] = new_resource

            else:
                # s[resource][resource_idx] = new_resource
                s[resource][get_resource_idx(job, op, "setup", solution)] = new_resource
                s[resource][get_resource_idx(job, op, "intermediate", solution)] = new_resource
                s[resource][get_resource_idx(job, op, "teardown", solution)] = new_resource

    else:
        idx1, idx2 = random.sample(list(range(len(s["order"]))), 2)
        max_idx = max(idx1, idx2)
        min_idx = min(idx1, idx2)
        ov = s["order"]
        s["order"] = [int(elem) for elem in np.concatenate((ov[:min_idx], [ov[max_idx]], ov[min_idx+1:max_idx], [ov[min_idx]], ov[max_idx+1:]))]

    return s


def random_solution(this_instance_data):
    order = []
    machines = []
    analysts = []
    jop = []

    for job_info in this_instance_data['jobs']:
        job = job_info["job"]
        for test in job_info["tests"]:
            op = test["test"]
            machine_resource = np.random.choice(test["suitable_machines"], 1)[0]
            analyst_resource = np.random.choice(test["suitable_analysts"], 1)[0]
            for task in ["setup", "intermediate", "teardown"]:
                order.append(job)
                machines.append(machine_resource)
                analysts.append(analyst_resource)
                jop.append((job, op, task, "analyst"))

    random.shuffle(order)
    solution = {"order": order, 'machines': machines, 'analysts': analysts, 'data': this_instance_data, "jop": jop}
    solution = fix_order(solution, list(range(1, 8)))
    solution = fix_order(solution, list(range(1, 8)))

    verify_solution(decode(solution), this_instance_data)
    return solution


def compatible_experiences(instance, method, past_data):
    indexes = []
    idx = 0
    for result in past_data[instance][method]:
        if np.all([result["parameters"][parameter] == param["tabu_search"][parameter] for parameter in ["max_steps", "ls_steps", "tabu_list_size"]]):
            indexes.append(idx)
            idx += 1
    return indexes


def all_runs_done(instance, method, past_data):
    desired_reps = 1
    if method[:4] == "TABU":
        desired_reps = param["number_of_repetitions"]

    if method[:4] == "HEUR":
        method = method.split("_")[1]
    if instance in past_data.keys() and method in past_data[instance].keys():
        if method[:4] == "TABU":
            indexes = compatible_experiences(instance, method, past_data)  # + compatible_experiences(instance, method_og, past_data)
            print("{}/{} runs for {} using {}".format(len(indexes), param["number_of_repetitions"], instance, method))
            return len(indexes) >= param["number_of_repetitions"]

        else:  # HEUR and CPLEX
            print("{}/{} runs for {} using {}".format(len(past_data[instance][method]), desired_reps, instance, method))
            return len(past_data[instance][method]) >= desired_reps

    else:  # method in past_data[instance].keys():
        print("No execution of method {} or no instance {}".format(method, instance))
    return False


def check_size_divergence(df_schedule):
    if len(df_schedule.loc[df_schedule.resource_type == "machine"].job) != len(df_schedule.loc[df_schedule.resource_type == "analyst"].job):
        print("#machines: {}, #analysts: {}".format(len(df_schedule.loc[df_schedule.resource_type == "machine"].job), len(df_schedule.loc[df_schedule.resource_type == "analyst"].job)))
        print("")


def run():
    ex_parameters = settings['experiments']
    path_instances = os.path.join(path_project, settings['paths']['instances'])
    path_results = os.path.join(path_project, settings['paths']['results'])

    saved_schedules = {}
    past_data = read_past_results(path_results)
    force_run = True
    force_run_heur = True
    param["print_plot"] = True

    for i in range(1):  #range(param["number_of_repetitions"]):
        print("========= REPETITION ", i)
        # Iterate over desired problems
        for n_jobs in ex_parameters['n_jobs']:
            for n_machines in ex_parameters['n_machines']:
                for n_workers in ex_parameters['n_workers']:
                    for n_sample_types in ex_parameters['n_sample_types']:
                        for flexibility in ex_parameters['flexibility']:
                            for i_replication in ex_parameters['replications']:
                                instance = 'ex_j' + str(n_jobs) + '_' + \
                                           'm' + str(n_machines) + '_' + \
                                           'w' + str(n_workers) + '_' + \
                                           's' + str(n_sample_types) + '_' + \
                                           'f' + str(flexibility) + '_' + \
                                           'r' + str(i_replication)
                                filename = instance + '.json'
                                saved_schedules[instance] = {}
                                print('\nStart instance ', instance)

                                # Load instance data
                                with open(os.path.join(path_instances, filename)) as json_file:
                                    instance_data = json.load(json_file)

                                # ------------------------------------------------------------------------------------------------------------
                                # HEURISTIC
                                for heuristic_type in ["M-LQ", "SPT", "LPT", "M-LAST"]:
                                    # Run heuristic
                                    heuristic_key = "HEURISTIC_" + heuristic_type
                                    # if not all_runs_done(instance, heuristic_key, past_data):
                                    if force_run_heur or (not all_results_done(instance, heuristic_key, settings, past_data)):
                                        rtime, df_schedule, instance_data, problem_data = heuristic(heuristic_type, instance_data)
                                        verify_solution(df_schedule, instance_data)

                                        df_schedule = df_schedule.replace("worker", "analyst")
                                        results = drc_eval(df_schedule)
                                        results["algorithm"] = heuristic_key
                                        results["time"] = rtime
                                        save_result(df_schedule, instance_data, path_results, results, instance, settings)

                                        print(heuristic_key, ": ", results["total_completion_time"])
                                        figure_obj = results["total_completion_time"]

                                    else:
                                        print("Using cached solution of " + heuristic_type)
                                        df_schedule, heur_info, instance_data_heur = get_latest_result(instance, heuristic_key, settings, past_data)

                                        results = drc_eval(df_schedule)
                                        figure_obj = results["total_completion_time"]

                                    df_schedule = df_schedule.replace("worker", "analyst")

                                    if param["print_plot"]:
                                        sched_fig, sched_ax = plot_schedule(df_schedule, legend="heuristic", verbose=False)
                                        sched_ax.set_title(instance + "\nHeuristic " + heuristic_type + ", completion time: " + str(figure_obj))
                                        sched_fig.show()

                                    check_size_divergence(df_schedule)
                                    df_schedule = df_schedule.replace("worker", "analyst")

                                # ------------------------------------------------------------------------------------------------------------
                                # TABU SEARCH (starting on the heuristic solution)

                                # Run heuristic + TABU
                                heuristic_tabu_key = "tabu_search"
                                if force_run or (not all_results_done(instance, heuristic_tabu_key, settings, amount=settings["parameters"]["number_of_repetitions"], initial_solution=heuristic_type)):
                                    verify_solution(df_schedule, instance_data)
                                    tabu_schedule, tabu_info = tabu_search(df_schedule, instance_data, instance)
                                    verify_solution(tabu_schedule, instance_data)

                                    tabu_results = drc_eval(tabu_schedule)
                                    tabu_results["algorithm"] = heuristic_tabu_key
                                    tabu_results["time"] = tabu_info["time"]
                                    tabu_results["unique_visited"] = tabu_info["unique_visited_solutions"]
                                    tabu_results["unique_generated"] = tabu_info["unique_generated_solutions"]
                                    tabu_results["iterations"] = tabu_info["total_steps"]
                                    tabu_results["converged_iterations"] = tabu_info["total_steps_no_imp"]
                                    tabu_results['initial_solution'] = heuristic_type
                                    save_result(tabu_schedule, instance_data, path_results, tabu_results, instance, settings)

                                    figure_obj = tabu_results["total_completion_time"]

                                else:
                                    print("Using cached solution of " + heuristic_tabu_key)
                                    tabu_schedule, tabu_info, instance_data2 = get_latest_result(instance, heuristic_tabu_key, settings)
                                    tabu_results = drc_eval(tabu_schedule)
                                    figure_obj = tabu_results["total_completion_time"]
                                #
                                if param["print_plot"]:
                                    tabu_fig, tabu_ax = plot_schedule(tabu_schedule, legend='decoded')
                                    tabu_ax.set_title(instance + "\nHeuristic " + heuristic_type + " + TABU" + ", completion time: " + str(figure_obj))
                                    tabu_fig.show()

                                # ------------------------------------------------------------------------------------------------------------
                                # Run TABU (from random initial solution)
                                tabu_key = "tabu_search"
                                if force_run or (not all_results_done(instance, tabu_key, settings, amount=settings["parameters"]["number_of_repetitions"], initial_solution="random")):
                                    random_initial_solution = decode(random_solution(instance_data))

                                    tabu_random_solution, tabu_random_info = tabu_search(random_initial_solution, instance_data, instance)

                                    tabu_random_results = drc_eval(tabu_random_solution)
                                    tabu_random_results["algorithm"] = tabu_key
                                    tabu_random_results["time"] = tabu_random_info["time"]
                                    tabu_random_results["unique_visited"] = tabu_random_info["unique_visited_solutions"]
                                    tabu_random_results["unique_generated"] = tabu_random_info["unique_generated_solutions"]
                                    tabu_random_results["iterations"] = tabu_random_info["total_steps"]
                                    tabu_random_results["converged_iterations"] = tabu_random_info["total_steps_no_imp"]
                                    tabu_random_results['initial_solution'] = "random"
                                    save_result(tabu_random_solution, instance_data, path_results, tabu_random_results, instance, settings)

                                    figure_obj = tabu_random_results["total_completion_time"]

                                else:
                                    print("Using cached solution to " + "TABU")
                                    tabu_random_solution, tabu_random_info, instance_data2 = get_latest_result(instance, tabu_key, settings)
                                    tabu_results = drc_eval(tabu_random_solution)
                                    figure_obj = tabu_results["total_completion_time"]
                                #
                                if param["print_plot"]:
                                    rand_fig, rand_ax = plot_schedule(tabu_random_solution, legend="random")
                                    rand_ax.set_title(instance + "\nTabu from random" + ", completion time: " + str(figure_obj))
                                    rand_fig.show()

        force_run_heur = False


if __name__ == "__main__":
    run()
