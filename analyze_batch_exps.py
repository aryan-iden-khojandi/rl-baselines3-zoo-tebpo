import os
import csv

import pandas as pd

EXPERIMENT_RESULTS_PATH = "experimental_results"
TRPO_RESULTS_PREFIX = "TRPO_ANALYSIS_"
TEBPO_RESULTS_PREFIX = "TEBPO_MC_ANALYSIS"
EVALUATION_PREFIX = "evaluation_"

num_experiments = 100

analysis_results = pd.DataFrame(columns=["TRPO Objective", "TEBPO Objective", "Ground Truth"])


def read_surr_obj_from_file(experiment_index, filename):

    with open(os.path.join(EXPERIMENT_RESULTS_PATH, str(experiment_index), filename), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            _, surr_obj_val = row

    return float(surr_obj_val)


def read_evaluation_val_from_file(experiment_index, filename):

    with open(os.path.join(EXPERIMENT_RESULTS_PATH, str(experiment_index), filename), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = [_ for _ in reader]
        _, obj_mean = rows[0]
        _, obj_std_err = rows[1]

    return float(obj_mean), float(obj_std_err)


for i in range(0, num_experiments):

    trpo_files = []
    tebpo_files = []
    evaluation_files = []

    for filename in os.listdir(os.path.join(EXPERIMENT_RESULTS_PATH, str(i))):

        if filename.startswith(TRPO_RESULTS_PREFIX):
            trpo_files.append(filename)

        elif filename.startswith(TEBPO_RESULTS_PREFIX):
            tebpo_files.append(filename)

        elif filename.startswith(EVALUATION_PREFIX):
            evaluation_files.append(filename)

        else:
            print("File {} did not match any relevant formats; skipping it.".format(filename))

    # Take just one surrogate-objective value for each method for now
    trpo_surr_obj_val = read_surr_obj_from_file(i, trpo_files[0])
    tebpo_surr_obj_val = read_surr_obj_from_file(i, tebpo_files[0])

    obj_mean_real_1, obj_std_err_real_1 = read_evaluation_val_from_file(i, evaluation_files[0])
    obj_mean_real_2, obj_std_err_real_2 = read_evaluation_val_from_file(i, evaluation_files[1])

    analysis_results.loc[i] = [trpo_surr_obj_val, tebpo_surr_obj_val, obj_mean_real_2 - obj_mean_real_1]
