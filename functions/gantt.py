import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np

plt.style.use("seaborn-whitegrid")
#plt.style.use("seaborn-deep")
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'
#plt.rcParams['grid.color'] = '0'
plt.rcParams['axes.edgecolor'] = '0'
plt.rcParams['text.color'] = '0'
plt.rcParams['xtick.color'] = '0'
plt.rcParams['ytick.color'] = '0'
plt.rcParams['axes.labelcolor'] = '0'
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.dpi'] = 100
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['grid.linewidth'] = 0.5


def plot_schedule(df_schedule, verbose=1, legend=True):
    cmap_1 = cm.get_cmap('tab20')
    cmap_2 = cm.get_cmap('tab20b')
    colors = []


    for i in list(np.arange(0, 1, 1/20)):
        colors.append(cmap_1(i))

    for i in list(np.arange(0, 1, 1/20)):
        colors.append(cmap_2(i))

    job_op = df_schedule.loc[:, ['job', 'operation']].drop_duplicates().set_index(['job', 'operation'])
    job_op['color'] = [()]*len(job_op)
    ix_color = 0
    ix_job = 0

    for job, op in zip(job_op.index.get_level_values(0), job_op.index.get_level_values(1)):
        if job != ix_job:
            ix_job = job
            ix_color += 1
            if ix_color == len(colors):
                ix_color = 0

        job_op.loc[(job, op), 'color'] = colors[ix_color]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    df_resources = df_schedule[['resource_type', 'resource']].drop_duplicates().sort_values(
        by=['resource_type', 'resource'])

    df_schedule.replace('analyst', 'worker', inplace=True)
    df_resources.replace('analyst', 'worker', inplace=True)

    df_resources['ix'] = np.arange(0, len(df_resources))
    df_resources['pos'] = np.arange(0.5, len(df_resources) * 0.5 + 0.5, 0.5)
    df_resources.set_index(['resource_type', 'resource'], inplace=True)

    plot_jobs = df_schedule['job'].unique()
    plot_jobs.sort()
    for job in plot_jobs:
        job_schedule = df_schedule.loc[df_schedule['job'] == job]
        for i_operation in job_schedule['operation'].unique():
            operation_schedule = job_schedule.loc[job_schedule['operation'] == i_operation]
            ix_tasks = {resource_key: 0 for resource_key in list(df_resources.index.get_level_values(0).unique())}
            for task in operation_schedule.iterrows():
                resource_type = task[1]['resource_type']
                resource = task[1]['resource']
                start = task[1]['start']
                end = task[1]['end']
                ix_tasks[resource_type] += 1

                color = job_op.loc[(job, 1), 'color']
                # color = job_op.loc[(job, task[1]['operation']), 'color']

                pos = df_resources.loc[(resource_type, resource), 'pos']
                ax.barh(pos, end - start, left=start, height=0.3, align='center',
                        edgecolor='black', color=color, alpha=0.8)

                if verbose > 0:
                    ax.text(end - (end - start)/1.5, pos-0.105, str(job)+'\n'+str(i_operation)+'\n'+str(ix_tasks[resource_type]), fontsize=8)

    ylabels = [str(row[1]['resource_type']) + ' ' + str(row[1]['resource']) for row in
               df_resources.reset_index().iterrows()]
    ypos = df_resources['pos'].tolist()
    plt.yticks(ypos, ylabels)

    legend_elements = []
    for job in df_schedule['job'].unique():
        color = job_op.loc[(job,), 'color'].values[0]

        legend_elements.append(
            Patch(facecolor=color, edgecolor='black', label='Job #' + str(job))
        )

    if legend:
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.15, 0.5))
    fig.subplots_adjust(right=0.80, bottom=0.15)

    ax.set_xlabel('Time')
    # plt.show()

    return fig, ax
