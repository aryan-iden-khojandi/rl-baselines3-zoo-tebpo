n_evals = 1000
n_steps = 10000
lams = [0.5, 0.9, 0.97, 0.98, 0.99, 1.]
envs = ["LunarLander-v2"]

eval_targets = expand("results/{env}/seed-{seed}/{eval_type}-{lam}.csv",
                 # eval_type=["evaluation", "policy_objective"],
                 eval_type=["evaluation"],
                 env=envs,
                 seed=range(10),
                 lam=lams)

obj_targets = expand("results/{env}/seed-{seed}/policy_objective-{algo}.csv",
                        # eval_type=["evaluation", "policy_objective"],
                        algo=["trpo", "tebpo_trpo", "tebpo_mc"],
                        env=envs,
                        seed=range(10))


rule targets:
    input:
        eval_targets + obj_targets


rule objs:
    input:
        obj_targets


rule evals:
    input:
        eval_targets


rule model_combinations:
    output:
        "models/{env}/model-{lam}.pkl"
    shell:
        "python3 scripts/combine_models.py"
        "  models/{wildcards.env}/init_model.pkl"
        "  models/{wildcards.env}/final_model.pkl"
        "  {output}"
        "  --lam {wildcards.lam}"


rule evaluation:
    input:
        "models/{env}/model-{lam}.pkl"
    output:
        "results/{env}/seed-{seed}/evaluation-{lam}.csv"
    shell:
        "python3 evaluate.py"
        "  --algo trpo"
        "  --seed {wildcards.seed}"
        "  --env {wildcards.env}"
        "  --output {output}"
        "  --n-evaluations {n_evals}"
        "  --init-policy-file {input}"


rule policy_objective:
    input:
        init_model="models/{env}/model-1.0.pkl",
        all_models=[f"models/{{env}}/model-{lam}.pkl" for lam in lams]
    output:
        "results/{env}/seed-{seed}/policy_objective-{algo}.csv"
    shell:
        "python3 get_policy_objective.py"
        "  --algo {wildcards.algo}"
        "  --seed {wildcards.seed}"
        "  --env {wildcards.env}"
        "  --init-policy-file {input.init_model}"
        "  --eval-policies {input.all_models}"
        "  -params n_steps:{n_steps} gae_lambda:1 gamma:1 normalize_advantage:False"
        "  --n-timesteps {n_steps}"
        "  --output {output}"
