def drc_eval(df_schedule, solution={}, objective=""):
    results = {'c_max': round(df_schedule['end'].max(), 6),
               'total_completion_time': round(sum(df_schedule.groupby(['job']).max().end), 6)}

    if objective == 'total_tardiness':
        results['total_tardiness'] = solution._objective
    elif objective == 'maximum_lateness':
        results['maximum_lateness'] = solution._objective
    elif objective == 'total_completion_time':
        results['total_completion_time'] = solution._objective


    return results
