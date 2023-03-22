import csv
import os
from copy import copy
import pandas as pd


def read_policy_objective_file(results_path, algo, seed):
    df = pd.DataFrame(columns=['algo', 'seed', 'policy', 'obj', 'kl'])
    canonical_row_values = {
        'algo': algo,
        'seed': seed
    }

    filename = os.path.join(results_path,
                            "seed-{seed}".format(seed=seed),
                            "policy_objective-{algo}.csv".format(algo=algo)
                            )
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    for row in rows:
        new_df_row = row
        new_df_row['obj'] = float(new_df_row['obj'])
        new_df_row['kl'] = float(new_df_row['kl'])
        new_df_row.update(canonical_row_values)
        df = df.append(new_df_row, ignore_index=True)

    return df


def read_eval_file(results_path, seed, lam):

    df = pd.DataFrame(columns=['seed', 'lam', 'mean', 'sd', 'n'])
    canonical_row_values = {
        'seed': seed,
        'lam': lam
    }

    filename = os.path.join(results_path,
                            "seed-{seed}".format(seed=seed),
                            "evaluation-{lam}.csv".format(lam=lam))

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    for row in rows:
        new_df_row = copy(row)
        new_df_row['mean'] = float(new_df_row['mean'])
        new_df_row['sd'] = float(new_df_row['sd'])
        new_df_row['n'] = int(new_df_row['n'])
        new_df_row.update(canonical_row_values)
        df = df.append(new_df_row, ignore_index=True)

    return df


def create_result_dataframes(algos, lams, seeds, results_path='results/CartPoleIncentive-v1'):

    df_policy_objective_all = pd.DataFrame()
    df_eval_all = pd.DataFrame()

    for seed in seeds:
        for algo in algos:
            df_policy_objective = read_policy_objective_file(results_path, algo, seed)
            df_policy_objective_all = df_policy_objective_all.append(df_policy_objective)
        for lam in lams:
            df_eval = read_eval_file(results_path, seed, lam)
            df_eval_all = df_eval_all.append(df_eval)

    return df_policy_objective_all, df_eval_all


if __name__ == "__main__":

    algos = ["trpo", "tebpo_mc"]
    lams = [0.5, 0.9, 0.97, 0.98, 0.99, 1.]
    seeds = range(10)

    df_policy_objective, df_eval = create_result_dataframes(algos, lams, seeds)

    # For each (policy, lam) pair, look at the distribution across seeds to get mean and standard deviation
    df_policy_objective_means = df_policy_objective.groupby(['algo', 'policy'])['obj'].mean().rename('obj_mean')
    df_policy_objective_stds = df_policy_objective.groupby(['algo', 'policy'])['obj'].std().rename('obj_std')

    df_policy_objective_summary = pd.concat([df_policy_objective_means, df_policy_objective_stds], axis=1)

    df_eval_means = df_eval.groupby(['lam'])['mean'].mean().rename('eval_mean')
    df_eval_stds = df_eval.groupby(['lam'])['mean'].std().rename('eval_std')

    df_eval_summary = pd.concat([df_eval_means, df_eval_stds], axis=1)
    reference_policy_eval_mean = df_eval_summary.query('lam == 1.0')['eval_mean'].tolist()[0]
    df_eval_summary['obj_improvement'] = reference_policy_eval_mean - df_eval_summary['eval_mean']

    print(df_policy_objective_summary)
    print(df_eval_summary)