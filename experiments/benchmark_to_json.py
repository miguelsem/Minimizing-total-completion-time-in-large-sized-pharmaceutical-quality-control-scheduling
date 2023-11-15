import pandas as pd
import os
from math import isnan
from collections import defaultdict
from experiments.new_heuristic import heuristic


def process_benchmark(benchmark_file="abz5"):
    file_source = "C:\\src\\drc-sched-3\\instances_txt"
    data = pd.read_csv(os.path.join(file_source, benchmark_file), delimiter="\t", header=None)[0]

    file_out = {}
    jobs = []
    n_analysts = 100
    suitable_analysts = list(range(1, n_analysts+1))
    for row in data:
        if row[0] != "#":
            # print(row.split(" "))
            row = [int(val) for val in row.split(" ") if val != ""]
            if file_out == {}:
                n_analysts = row[0]*10
                file_out = {"n_jobs": row[0], "n_analysts": n_analysts, "n_machines": row[1], "jobs": []}
            else:
                tests = []
                for machine_id, machine_time in [(row[i*2], row[int(i*2+1)]) for i in range(int(len(row)/2))]:
                    tests.append({
                        "test": len(tests)+1, "machine_time": machine_time, "analyst_times":
                            [
                                {"step": "setup", "start": 0, "end": 0},
                                {"step": "intermediate", "start": machine_time*.1, "end": machine_time*.1},
                                {"step": "teardown", "start": machine_time*.2, "end": machine_time*.2}
                            ],
                        "suitable_analysts": suitable_analysts,
                        "suitable_machines": [machine_id+1]
                        })

                jobs.append({
                    "job": len(jobs)+1,
                    "product": 0,
                    "tests": tests
                    })

    file_out["jobs"] = jobs

    # Get heuristic and optimal results
    baselines = pd.read_excel("C:\\src\\drc-sched-3\\experiments\\BaselineResults.xlsx").set_index("filename")
    if not benchmark_file in baselines.keys():
        # Get heuristic solution
        baselines.loc[benchmark_file, "LPT"] = longest_job_heuristic(file_out)
        # Save baseline excel
        baselines.to_excel("C:\\src\\drc-sched-3\\experiments\\BaselineResults.xlsx")

    file_out["baselines"] = {key: val for key, val in baselines.loc[benchmark_file].items() if not isnan(val)}


    return file_out


def process_benchmark_flex(bechnmark_file="MK01.txt", flex=True):
    file_source = "instances"
    data = pd.read_csv(os.path.join(file_source, bechnmark_file), delimiter="\t", header=None)[0]

    file_out = {}
    jobs = []
    n_analysts = 100
    suitable_analysts = list(range(1, n_analysts+1))
    for row in data:
        if row[0] != "#":
            # print(row.split(" "))
            row = [int(val) for val in row.split(" ") if val != ""]
            if file_out == {}:
                n_analysts = row[0]*10
                file_out = {"n_jobs": row[0], "n_analysts": n_analysts, "n_machines": row[1], "jobs": []}
            else:

                offset = 0
                if flex:
                    offset = 1

                tests = []
                mach_idx = 1
                for test in range(1, row[0]+1):
                    mach_n = row[mach_idx]
                    # machine_times = [(row[i*2+offset], row[int(i*2+1+offset)]) for i in range(int(len(row)/2))]:
                    machine_times = [row[mach_idx + 2 + i*2] for i in range(mach_n)]
                    machine_ids = [row[mach_idx + 1 + i*2] for i in range(mach_n)]
                    # print("IDs: ", machine_ids)
                    # print("Times: ", machine_times)
                    mach_idx += 1 + mach_n*2

                    machine_times_1 = list(map(lambda x: x*.1, machine_times))
                    machine_times_2 = list(map(lambda x: x*.2, machine_times))
                # for machine_id, machine_time in [(row[i*2+offset], row[int(i*2+1+offset)]) for i in range(int(len(row)/2))]:
                    tests.append({
                        "test": test, "machine_time": machine_times, "analyst_times":
                            [
                                {"step": "setup", "start": 0, "end": 0},
                                {"step": "intermediate", "start": machine_times_1, "end": machine_times_1},
                                {"step": "teardown", "start": machine_times_2, "end": machine_times_2}
                            ],
                        "suitable_analysts": suitable_analysts,
                        "suitable_machines": machine_ids
                        })

                jobs.append({
                    "job": len(jobs)+1,
                    "product": 0,
                    "tests": tests
                    })

    file_out["jobs"] = jobs
    return file_out


def longest_job_heuristic(data):
    # heuristic("LPT", data)  # ["M-LAST", "M-LQ", "SPT", "LPT"]:
    solution = {}
    for strategy in ["LPT"]:  #, "SPT"]:
        runtime, df, _, result = heuristic(strategy, data)
        solution[strategy] = result
    return result["c_max"]


if __name__ == '__main__':
    process_benchmark()