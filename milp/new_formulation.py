import os
import sys
import yaml
import json
from docplex.mp.model import Model
import pandas as pd
import matplotlib.pyplot as plt
from docplex.mp.relax_linear import LinearRelaxer


def milp(instance_data, params):
    ''' Sets '''

    # number of samples
    n = len(instance_data['jobs'])

    # number of analysts
    n_ana = instance_data['n_analysts']

    # set of jobs
    J = [j for j in range(1, n+1)]

    # set of operations
    O = [(j, i) for j in J for i in range(1, len(instance_data['jobs'][j-1]['tests'])+1)]

    # set of tasks
    Y = [(j, i, s) for j in J for i in range(1, len(instance_data['jobs'][j-1]['tests'])+1)
         for s in range(1, len(instance_data['jobs'][j-1]['tests'][i-1]['analyst_times'])+1)]


    # set of equipment suitable for processing the test
    K = [(j, i, k) for j, i in O for k in instance_data['jobs'][j-1]['tests'][i-1]['suitable_machines']]

    # set of analysts suitable for processing the test
    W = [(j, i, h) for j, i in O for h in instance_data['jobs'][j-1]['tests'][i-1]['suitable_analysts']]

    ''' Parameters '''

    # processing times
    p = {(j, i): instance_data['jobs'][j-1]['tests'][i-1]['machine_time'] for j, i in O}

    # release times
    r = {(j, i): instance_data['jobs'][j - 1]['release_time'] for j, i in O}

    # release times
    d = {(j): instance_data['jobs'][j - 1]['due_date'] for j, i in O}

    # Number of analyst interventions
    Ni = {(j, i): len(instance_data['jobs'][j-1]['tests'][i-1]['analyst_times']) for j, i in O}

    # Number of tests
    c = {j: len(instance_data['jobs'][j-1]['tests']) for j in J}

    # Number of tasks
    N = {(j, i): len(instance_data['jobs'][j-1]['tests'][i-1]['analyst_times']) for j, i in O}

    # Analyst start time
    rho_s = {(j, i, s): instance_data['jobs'][j-1]['tests'][i-1]['analyst_times'][s-1]['start']
        for j, i in O for s in range(1, N[j, i]+1)}

    # Analyst duration
    rho_d = {(j, i, s): (instance_data['jobs'][j-1]['tests'][i-1]['analyst_times'][s-1]['end'] -
                         instance_data['jobs'][j-1]['tests'][i-1]['analyst_times'][s-1]['start'])
             for j, i in O for s in range(1, N[j, i]+1)}

    ''' Tuning parameters '''

    M = 300

    ''' Variables '''

    x_dict = [(j, i, k) for j, i, k in K]

    t_dict = [(j, i) for j, i in O]

    T_dict = [(j) for j in J]

    beta_dict = [(j, i, j_, i_) for j, i in O for j_, i_ in O]

    alfa_dict = [(j, i, h) for j, i, h in W]

    gamma_dict = [(j, i, s, j_, i_, s_) for j, i, s in Y for j_, i_, s_ in Y]

    cj_dict = [(j) for j in J]

    ''' Model creation '''

    mdl = Model('DRC')

    mdl.set_time_limit(params['time_limit'])

    x = mdl.binary_var_dict(x_dict, name='x')

    t = mdl.continuous_var_dict(t_dict, lb=0, name='t')

    beta = mdl.binary_var_dict(beta_dict, name='beta')


    if params['dual_resource']:
        alfa = mdl.binary_var_dict(alfa_dict, name='alfa')

        gamma = mdl.binary_var_dict(gamma_dict, name='gamma')

    mdl.add_constraints(t[j, i] >= t[j, i-1] + p[j, i-1]
                        for j, i in O if i != 1)

    mdl.add_constraints(t[j, i] >= r[j, i]
                        for j, i in O)

    mdl.add_constraints(mdl.sum(x[j, i, k] for j_, i_, k in K if j_ == j and i_ == i) == 1
                        for j, i in O)

    mdl.add_constraints(t[j, i] >= t[j_, i_] + p[j_, i_] - (2 - x[j, i, k] - x[j_, i_, k_] + beta[j, i, j_, i_])*M
                        for j, i, k in K for j_, i_, k_ in K if k == k_ and not (i == i_ and j == j_))

    mdl.add_constraints(t[j_, i_] >= t[j, i] + p[j, i] - (3 - x[j, i, k] - x[j_, i_, k_] - beta[j, i, j_, i_])*M
                        for j, i, k in K for j_, i_, k_ in K if k == k_ and not (i == i_ and j == j_))


    if params['dual_resource']:
        mdl.add_constraints(mdl.sum(alfa[j, i, h] for j_, i_, h in W if j_ == j and i_ == i) == 1
                            for j, i in O)

        mdl.add_constraints(t[j, i] + rho_s[j, i, s] >= t[j_, i_] + rho_s[j_, i_, s_] + rho_d[j_, i_, s_] - (2 - alfa[j, i, h] - alfa[j_, i_, h_] + gamma[j, i, s, j_, i_, s_]) * M
                            for j, i, h in W for j_, i_, h_ in W if h == h_ and not (i == i_ and j == j_) for s in range(1, N[(j, i)]+1) for s_ in range(1, N[(j_, i_)]+1))

        mdl.add_constraints(t[j_, i_] + rho_s[j_, i_, s_] >= t[j, i] + rho_s[j, i, s] + rho_d[j, i, s] - (3 - alfa[j, i, h] - alfa[j_, i_, h_] - gamma[j, i, s, j_, i_, s_]) * M
                            for j, i, h in W for j_, i_, h_ in W if h == h_ and not (i == i_ and j == j_) for s in range(1, N[(j, i)]+1) for s_ in range(1, N[(j_, i_)]+1))

    if params['objective'] == 'c_max':
        c_max = mdl.continuous_var(lb=0, name='c_max')

        mdl.add_constraints(c_max >= t[j, c[j]] + p[j, c[j]] for j in J)

        mdl.minimize(c_max)
    elif params['objective'] == 'total_tardiness':
        T = mdl.continuous_var_dict(T_dict, lb=0, name='total_tardiness')

        mdl.add_constraints(T[j] >= t[j, c[j]] + p[j, c[j]] - d[j] for j in J)

        mdl.minimize(mdl.sum(T))
    elif params['objective'] == 'maximum_lateness':
        L_max = mdl.continuous_var(lb=0, name='maximum_lateness')

        mdl.add_constraints(L_max >= t[j, c[j]] + p[j, c[j]] - d[j] for j in J)

        mdl.minimize(L_max)
    elif params['objective'] == 'total_completion_time':
        mdl.minimize(mdl.sum(t[j, c[j]] + p[j, c[j]] for j in J))

    mdl.print_information()

    relaxed_model = LinearRelaxer.linear_relaxation(mdl, mdl)
    relaxed_model.set_time_limit(9999999999999)
    lp_solution = relaxed_model.solve()
    lp_solution_dict = lp_solution.as_name_dict()
    lb = lp_solution._objective

    if 'emphasis' in params.keys():
        mdl.parameters.emphasis.mip = params['emphasis']
    else:
        mdl.parameters.emphasis.mip = 0

    solution = mdl.solve(log_output=params['log'])
    solution_found = False

    print("Details: ", mdl.solve_details)
    print("Status: ", mdl.solve_status)

    try:
        solution_dict = solution.as_name_dict()
        solution_found = True
    except:
        print("\n\nNo solution found: ", mdl.solve_details)

        solution = lp_solution
        solution_dict = lp_solution_dict


    df_schedule = pd.DataFrame(columns=['start', 'end', 'job', 'operation', 'task', 'resource', 'resource_type'])
    df_sched_ix = 0

    if solution_found:
        for ix_job, job in enumerate(instance_data['jobs']):
            for ix_test, test in enumerate(job['tests']):
                try:
                    start = round(solution_dict['t'+'_'+str(ix_job+1)+'_'+str(ix_test+1)], 8)
                except:
                    start = 0

                end = start + test['machine_time']

                machine_resource = -1
                for machine in test['suitable_machines']:
                    try:
                        if solution_dict['x'+'_'+str(ix_job+1)+'_'+str(ix_test+1)+'_'+str(machine)] == 1:
                            machine_resource = machine
                    except:
                        pass

                df_schedule.loc[df_sched_ix] = [start, end, ix_job+1, ix_test+1, 0, machine_resource, 'machine']
                df_sched_ix += 1

                if params['dual_resource']:
                    analyst_resource = -1
                    for analyst in test['suitable_analysts']:
                        try:
                            if solution_dict['alfa' + '_' + str(ix_job + 1) + '_' + str(ix_test + 1) + '_' + str(analyst)] == 1:
                                analyst_resource = analyst
                        except:
                            pass

                    for ix_task, analyst_time in enumerate(test['analyst_times']):
                        start_task = start + analyst_time['start']
                        end_task = start + analyst_time['end']

                        df_schedule.loc[df_sched_ix] = [start_task, end_task, ix_job+1, ix_test+1, ix_task, analyst_resource, 'analyst']

                        df_sched_ix += 1

    print(solution._solve_details)

    return solution, df_schedule, mdl, lb


if __name__ == '__main__':
    path_project = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))

    sys.path.append(path_project)

    from functions.gantt import plot_schedule
    from functions.evaluation import drc_eval

    path_settings = os.path.join(path_project, 'settings.yml')

    settings = yaml.load(open(path_settings, encoding='UTF-8'), yaml.FullLoader)
    params = settings['params_milp']
    print(params['time_limit'])
    params['time_limit'] = 60
    params['objective'] = 'total_completion_time'
    path_instances = os.path.join(path_project, settings['paths']['instances'])

    dual_resource = True

    instances = ['ex_j10_m7_w3_s3_f0.3_r0']
    # instances = ['ex_j70_m7_w3_s3_f0.3_r0']

    instance_file = str(instances[0]) + '.json'
    with open(os.path.join(path_instances, instance_file)) as json_file:
        instance_data = json.load(json_file)

    solution, df_schedule, mdl, lb = milp(instance_data, params)

    results = drc_eval(df_schedule, solution, params['objective'])
    gap = solution.solve_details.mip_relative_gap
    best_bound = solution.solve_details.best_bound
    time = solution.solve_details._time
    status = solution.solve_details._solve_status

    results['gap'] = gap
    results['best_bound'] = best_bound
    results['time'] = time
    results['status'] = status
    if pd.isna(gap):
        results['status'] = 'only lp relaxation'
    results['solver'] = 'CPLEX MILP'
    results['lb'] = lb

    for measure in results.keys():
        print(measure + ': ' + str(results[measure]))

    fig, ax = plot_schedule(df_schedule)
    plt.show()

    mdl.print_information()

    n_cols = solution.solve_details.columns
    n_constraints = mdl.number_of_constraints
    n_nonzeros = solution.solve_details._linear_nonzeros
    n_variables = mdl.number_of_variables

    density = n_nonzeros / (n_cols * n_constraints)
