from functions.gantt import plot_schedule
import matplotlib.pyplot as plt
import numpy as np


def verify_solution(df, data):
    all_results = []
    # all operations on same machine
    correct_machine = []
    job_op_combo = list(set([(row.job, row.operation) for index, row in df.loc[:, ["job", "operation"]].iterrows()]))
    for job, op in job_op_combo:
        values = df.loc[[a and b and c for a, b, c in zip(df.resource_type=="machine", df.job==job, df.operation==op)], "resource"].values
        result = all([value == values[0] for value in values])
        if not result:
            print("job {}, op {}, values {}".format(job, op, values))
        correct_machine.append(result)
    all_results.append(all(correct_machine))

    # correct total operations number
    n_jobs = len(set(df.job))
    all_results.append(df.loc[df["resource_type"] == "machine"].shape[0] == sum([len(jobs["tests"]) for jobs in data["jobs"]]) * 3)

    # 3 tasks per operation
    correct_number_tasks = []
    for job in data["jobs"]:
        for test in job["tests"]:
            correct_number_tasks.append(len(test["analyst_times"]) == 3)
    all_results.append(all(correct_number_tasks))

    # no temporal overlapping between preceding
    correct_order_of_tasks = []
    correct_order_of_operations = []
    precision = 0.0001
    for job in data["jobs"]:
        # Check for tasks
        for test in job["tests"]:
            df_subset = df.loc[[a and b and c for a, b, c in zip(df.job == job["job"], df.operation == test["test"], df.resource_type == "machine")]]
            t12 = df_subset.loc[df_subset.task == "setup", "end"].iloc[0] - precision < df_subset.loc[df_subset.task == "intermediate", "start"].iloc[0]
            t23 = df_subset.loc[df_subset.task == "intermediate", "end"].iloc[0] - precision < df_subset.loc[df_subset.task == "teardown", "start"].iloc[0]
            result = t12 and t23
            correct_order_of_tasks.append(result)
            if (not result):
                print(df_subset)

        # Check for operations
        n_ops = len(job["tests"])
        for opi in range(1, n_ops):
            task_end = df.loc[(df.job == job["job"]) & (df.operation == opi) & (df.resource_type == "machine") & (df.task == "teardown"), "end"].iloc[0]
            next_task_start = df.loc[(df.job == job["job"]) & (df.operation == (opi+1)) & (df.resource_type == "machine") & (df.task == "setup"), "start"].iloc[0]
            result = (task_end - precision) < next_task_start
            correct_order_of_operations.append(result)
            if (not result):
                print("Operations out of order")
    all_results.append(all(correct_order_of_tasks))
    all_results.append(all(correct_order_of_operations))

    # same number of operations on both resources
    all_results.append(df.loc[df["resource_type"] == "machine"].shape[0] == df.loc[df["resource_type"] == "analyst"].shape[0])

    # See if all resources on correct equipment
    correct_equipment = []
    for index, row in df.iterrows():
        correct_equipment.append(row.resource in data["jobs"][row.job-1]["tests"][row.operation-1]["suitable_"+row.resource_type+"s"])
    all_results.append(all(correct_equipment))

    # Correct number of machines and analysts
    all_results.append(len(set(df.loc[df.resource_type == "machine", "resource"])) <= data["n_machines"])
    all_results.append(len(set(df.loc[df.resource_type == "analyst", "resource"])) <= data["n_analysts"])

    # Correct ORDER of operations in same machines
    df = df.sort_values(by=["start"])
    # for resource_type in ["machine", "analyst"]:
    all_in_a_row = []
    for resource in set(df.loc[df.resource_type == "machine", "resource"]):
        resource_list = df.loc[[a == "machine" and b == resource for a, b in zip(df.resource_type, df.resource)]]
        jop = [(row.job, row.operation) for index, row in resource_list.iterrows()]
        for job_combo in set(jop):
            indexes = np.where([j == job_combo for j in jop])[0]
            result = (indexes[0]+2) == (indexes[1]+1) == indexes[2]
            all_in_a_row.append(result)
    all_results.append(all(all_in_a_row))

    # no temporal overlapping when sharing resource
    p = 0.000001
    no_overlapping = []
    for resource_type in ["machine", "worker"]:
        for resource in list(set(df.loc[df.resource_type == resource_type, "resource"])):
            subset = df.loc[(df.resource_type == resource_type) & (df.resource == resource), ["start", "end"]]
            subset = subset.sort_values(["start"])
            no_overlapping.append(np.all([subset.iloc[i+1, 0]+p >= subset.iloc[i, 1] for i in range(len(subset)-1)]))
    all_results.append(all(no_overlapping))

    veredict = all(all_results)
    if not veredict:
        plot_schedule(df)
        plt.show()
        print("Error!")
        # assert(False)
    return veredict

