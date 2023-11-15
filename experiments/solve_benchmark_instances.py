import os
import pandas as pd
from experiments.benchmark_to_json import process_benchmark
from experiments.new_heuristic import heuristic

instances_folder = "C:\\src\\drc-sched-3\\instances_txt"
for file in os.listdir(instances_folder):
    out = process_benchmark(file)


