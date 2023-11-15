import os
import sys
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime
import pickle
import time as ttime

from functions.gantt import plot_schedule
from milp.new_formulation import milp
from functions.evaluation import drc_eval

path_project = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
settings = yaml.load(open(os.path.join(path_project, "settings.yml"), encoding='UTF-8'), Loader=yaml.FullLoader)
paths = settings["paths"]
param = settings["parameters"]

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('experiments')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('experiments.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh.setFormatter(log_formatter)
ch.setFormatter(log_formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Scheduler is gonna get started...')


milp_base_params = settings['params_milp']

df_results = pd.DataFrame(columns=[
    'c_max',
    'total_completion_time',
    'solver',
    'heuristic_type',
    'gap',
    'best_bound',
    'lb',
    'status',
    'time',
    'n_jobs',
    'n_machines',
    'n_workers',
    'n_sample_types',
    'flexibility',
    'replication',
    "id"
])

saved_schedules = {}

milp_params = milp_base_params
milp_limit_day = 60*60
milp_limit_week = 60*60*12
milp_params['log'] = False

exp_settings = settings['experiments']
ix_results = 0

results_filename = 'results_CPLEX_'+datetime.datetime.now().strftime('%y%m%d-%H%M')+'.csv'
schedules_filename = 'milp_results_'+datetime.datetime.now().strftime('%y%m%d-%H%M')+'.pkl'

for n_jobs in exp_settings['n_jobs']:
    for n_machines in exp_settings['n_machines']:
        for n_workers in exp_settings['n_workers']:
            for n_sample_types in exp_settings['n_sample_types']:
                for flexibility in exp_settings['flexibility']:
                    for i_replication in exp_settings['replications']:
                        logger.info('Evaluating instances for:')
                        logger.info('- n_jobs = '+str(n_jobs))
                        logger.info('- n_machines = '+str(n_machines))
                        logger.info('- n_sample_types = '+str(n_sample_types))
                        logger.info('- n_workers = '+str(n_workers))
                        logger.info('- flexibility = '+str(flexibility))
                        logger.info('-- replication = '+str(i_replication))

                        instance = 'ex_j' + str(n_jobs) + '_' + \
                                   'm' + str(n_machines) + '_' + \
                                   'w' + str(n_workers) + '_' + \
                                   's' + str(n_sample_types) + '_' + \
                                   'f' + str(flexibility) + '_' + \
                                   'r' + str(i_replication)
                        filename = instance + '.json'
                        saved_schedules[instance] = {}

                        print("\n Instance: ", instance)
                        with open(os.path.join(paths["instances"], filename)) as json_file:
                            instance_data = json.load(json_file)

                        milp_params['time_limit'] = milp_limit_day
                        if n_jobs == 10:
                            milp_params['time_limit'] = milp_limit_day
                        elif n_jobs >= 70:
                            milp_params['time_limit'] = milp_limit_week

                        milp_replications = 1
                        for mip_emphasis in [0]:
                            logger.info('Starting MILP')
                            milp_params['emphasis'] = mip_emphasis
                            solution, df_schedule, mdl, lb = milp(instance_data, milp_params)

                            results = drc_eval(df_schedule)
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

                            this_date = datetime.date.today()
                            id = "{}_{}_{}_{}".format(this_date.year, this_date.month, this_date.day, str(round(ttime.time()) % 3600 * 24))

                            df_results.loc[ix_results] = [
                                results['c_max'],
                                results['total_completion_time'],
                                results['solver'],
                                np.nan,
                                results['gap'],
                                results['best_bound'],
                                results['lb'],
                                results['status'],
                                results['time'],
                                n_jobs,
                                n_machines,
                                n_workers,
                                n_sample_types,
                                flexibility,
                                i_replication,
                                id
                            ]

                            logger.info(df_results.iloc[-1])
                            df_results.to_csv(paths["results"] + results_filename)

                            saved_schedules[instance][results['solver']] = df_schedule

                            plot_schedule(df_schedule)
                            plt.title("CPLEX " + instance + " -> " + str(round(best_bound, 2)) + "\n")
                            plt.show()

                            print(solution.solve_details)

                            ix_results += 1


pickle.dump(saved_schedules, open(schedules_filename, 'wb'))
