import os
import pickle
import pandas as pd
import yaml
import datetime
import re


path_project = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
settings = yaml.load(open(os.path.join(path_project, "settings.yml"), encoding='UTF-8'), Loader=yaml.FullLoader)
paths = settings["paths"]


def read_past_results(path=paths["results"], prename=""):
    data = pd.DataFrame()
    for file in [file for file in os.listdir(path) if file[-4:] == ".csv"]:
        if prename != "":
            if prename in file:
                df = pd.read_csv(os.path.join(path, file))
                data = pd.concat([data, df])

        else:
            df = pd.read_csv(os.path.join(path, file))
            data = pd.concat([data, df])
    return data


def get_file(filename, path=paths["results"]):
    with open(os.path.join(path, filename), 'rb') as f:
        data = pickle.load(f)
    return data


def get_latest_result(instance, algorithm, settings, past_results=read_past_results()):
    instance_ids = instance.split("_")
    j = int(float(re.sub("[a-z]", "", instance_ids[1])))
    w = int(float(re.sub("[a-z]", "", instance_ids[3])))
    f = float(re.sub("[a-z]", "", instance_ids[5]))
    r = int(float(re.sub("[a-z]", "", instance_ids[6])))

    subset = past_results.loc[
        (j == past_results.jobs) &
        (w == past_results.workers) &
        (f == past_results.flexibility) &
        (r == past_results.replication) &
        (algorithm == past_results.algorithm)]

    if subset.shape[0] > 0 and algorithm in settings["parameters"].keys():
        indexes = [True]*len(subset)
        for param_name, param_value in settings["parameters"][algorithm].items():
            if param_name in subset.columns:
                indexes = indexes & (subset[param_name] == param_value)
            else:
                return None, None, None

        results_out = subset.loc[indexes].iloc[-1]
        results_out_id = results_out["id"]
        for i in [2, 3]:
            if len(results_out_id.split("_")[i]) < 2:
                results_out_id = "_".join(results_out_id.split("_")[:i]) + '_0' + "_".join(results_out_id.split("_")[i:])
        if len(results_out_id.split("_")[-1]) < 6:
            seconds = results_out_id.split("_")[-1]
            seconds = (6 - len(seconds)) * "0" + seconds
            results_out_id = "_".join(results_out_id.split("_")[:-1]) + "_" + (6 - len(seconds)) * "0" + seconds

        solution_out = get_file("solution_" + results_out_id + ".pkl")
        # solution_out = get_file("solution_" + results_out["id"] + ".pkl")

        return solution_out["solution"], results_out, solution_out["data"]

    if subset.shape[0] > 0:  # Heuristic
        results_out = subset.iloc[-1]
        results_out_id = results_out["id"]
        for i in [1, 2, 3]:
            if len(results_out_id.split("_")[i]) < 2:
                results_out_id = "_".join(results_out_id.split("_")[:i]) + '_0' + "_".join(results_out_id.split("_")[i:])
        if len(results_out_id.split("_")[-1]) < 6:
            seconds = results_out_id.split("_")[-1]
            seconds = (6 - len(seconds)) * "0" + seconds
            results_out_id = "_".join(results_out_id.split("_")[:-1])+"_"+seconds

        solution_out = get_file("solution_" + results_out_id + ".pkl")
        # solution_out = get_file("solution_" + results_out["id"] + ".pkl")
        return solution_out["solution"], results_out, solution_out["data"]
    return None, None, None


def all_results_done(instance, algorithm, settings, past_results=read_past_results(), amount=1, same_parameters=True, initial_solution=""):
    if past_results.shape[0] == 0:
        print("No results")
        return False
    # Instance data
    instance_ids = instance.split("_")
    j = int(re.sub('\D', '', instance_ids[1]))
    w = int(re.sub('\D', '', instance_ids[3]))
    f = float(re.sub('^[a-z]', '', instance_ids[5]))
    r = int(re.sub('\D', '', instance_ids[6]))

    subset = past_results.loc[
        (j == past_results.jobs) &
        (w == past_results.workers) &
        (f == past_results.flexibility) &
        (r == past_results.replication) &
        (algorithm == past_results.algorithm)]

    if initial_solution != "":
        subset = subset.loc[[value for value in subset.loc[:, "initial_solution"] == initial_solution]]

    if same_parameters and subset.shape[0] > 0 and algorithm in settings["parameters"].keys():
        indexes = [True]*len(subset)
        for param_name, param_value in settings["parameters"][algorithm].items():
            if param_name in subset.columns:
                indexes = indexes & (subset[param_name] == param_value)
            else:
                print("No results")
                return False
        subset = subset.loc[indexes]

    print("Results: {} (needed {})".format(subset.shape[0], amount))
    return subset.shape[0] >= amount


def save_result(df, instance_data, results_path, result, instance, settings):
    this_date = datetime.date.today()
    month = this_date.month if len(str(this_date.month)) > 1 else "0" + str(this_date.month)
    day = this_date.day if len(str(this_date.day)) > 1 else "0" + str(this_date.day)
    seconds = datetime.datetime.now().second + datetime.datetime.now().minute*60 + datetime.datetime.now().hour*60*60
    seconds = (6-len(str(seconds)))*"0" + str(seconds)
    current_time = "{}_{}_{}_{}".format(this_date.year, month, day, seconds)

    # Save result
    result["id"] = current_time

    # Instance data
    instance_ids = instance.split("_")
    result["jobs"] = re.sub('\D', '', instance_ids[1])
    result["machines"] = instance_ids[2][1:]
    result["workers"] = instance_ids[3][1:]
    result["sample_types"] = instance_ids[4][1:]
    result["flexibility"] = instance_ids[5][1:]
    result["replication"] = instance_ids[6][1:]

    # Add settings parameters
    if result["algorithm"] in settings["parameters"].keys():
        for key in settings["parameters"][result["algorithm"]].keys():
            parameter = settings["parameters"][result["algorithm"]][key]
            result[key] = parameter

    # results.append(result)
    this_result = {}
    for key, value in result.items():
        this_result[key] = [value]

    pd.DataFrame.from_dict(this_result).to_csv(os.path.join(results_path, "result_"+current_time+".csv"))

    # Save solution
    if type(df) != type(pd.DataFrame()):
        print("Error")
    solution = {"name": "solution_" + current_time, "solution": df, "data": instance_data}
    with open(os.path.join(results_path, "solution_"+current_time+".pkl"), "wb") as file:
        pickle.dump(solution, file)
