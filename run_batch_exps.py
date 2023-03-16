import os

# GENERATE_REFERENCE_POLICY_CMD = \
#     "python train.py --algo trpo --env Hopper-v4 -params n_steps:49999 --env-kwargs terminate_when_unhealthy:False max_episode_steps:100"
# MOVE_REFERENCE_POLICY_CMD = "mv saved_models/saved_model_TRPO input_models/fixed_model_TRPO_Hopper_{i}"
# GENERATE_MIXED_POLICY_CMD = "python train.py --algo trpo --env Hopper-v4 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_Hopper_1 --fixed-policy-file input_models/fixed_model_TRPO_Hopper_2 --env-kwargs max_episode_steps:100 terminate_when_unhealthy:False"
# MOVE_MIXED_POLICY_CMD = "mv saved_models/saved_mixed_model_TRPO input_models/fixed_model_TRPO_Hopper_{i}"
# RUN_TRPO_ANALYSIS_ON_INIT_MIXED_POLICIES = "python train.py --algo trpo_analysis --env Hopper-v4 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_Hopper_1 --fixed-policy-file input_models/fixed_model_TRPO_Hopper_2 --env-kwargs max_episode_steps:100 terminate_when_unhealthy:False --experiment-index {exp_idx}"
# RUN_TEBPO_MC_ANALYSIS_ON_INIT_MIXED_POLICIES = "python train.py --algo tebpo_mc_analysis --env Hopper-v4 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_Hopper_1 --fixed-policy-file input_models/fixed_model_TRPO_Hopper_2 --env-kwargs max_episode_steps:100 terminate_when_unhealthy:False --experiment-index {exp_idx}"
# RUN_EVALUATION = "python evaluate.py --algo trpo --env Hopper-v4 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_Hopper_{model_idx} --fixed-policy-file input_models/fixed_model_TRPO_Hopper_{model_idx} --env-kwargs max_episode_steps:100 terminate_when_unhealthy:False --experiment-index {exp_idx}"

GENERATE_REFERENCE_POLICY_CMD = \
    "python train.py --algo trpo --env CartPole-v1 -params n_steps:49999"
MOVE_REFERENCE_POLICY_CMD = "mv saved_models/saved_model_TRPO input_models/fixed_model_TRPO_CartPole_{i}"
GENERATE_MIXED_POLICY_CMD = "python train.py --algo trpo --env CartPole-v1 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_CartPole_1 --fixed-policy-file input_models/fixed_model_TRPO_CartPole_2"
MOVE_MIXED_POLICY_CMD = "mv saved_models/saved_mixed_model_TRPO input_models/fixed_model_TRPO_CartPole_{i}"
RUN_TRPO_ANALYSIS_ON_INIT_MIXED_POLICIES = "python train.py --algo trpo_analysis --env CartPole-v1 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_CartPole_1 --fixed-policy-file input_models/fixed_model_TRPO_CartPole_2 --experiment-index {exp_idx}"
RUN_TEBPO_MC_ANALYSIS_ON_INIT_MIXED_POLICIES = "python train.py --algo tebpo_mc_analysis --env CartPole-v1 -params n_steps:49999 --init-policy-file input_models/fixed_model_CartPole_Hopper_1 --fixed-policy-file input_models/fixed_model_TRPO_CartPole_2 --experiment-index {exp_idx}"
RUN_EVALUATION = "python evaluate.py --algo trpo --env CartPole-v1 -params n_steps:49999 --init-policy-file input_models/fixed_model_TRPO_CartPole_{model_idx} --fixed-policy-file input_models/fixed_model_TRPO_CartPole_{model_idx} --experiment-index {exp_idx}"

num_exps = 100
num_tebpo_runs_per_exp = 2
# Add seed option for multiple runs within an experiment

for i in range(100, 100 + num_exps):

    os.system("mkdir experimental_results/{}".format(i))

    # Generate first reference policy
    os.system(GENERATE_REFERENCE_POLICY_CMD)

    # Move saved policy to input_models directory to serve as the 'init' policy
    os.system(MOVE_REFERENCE_POLICY_CMD.format(i=1))

    # Generate second reference policy
    os.system(GENERATE_REFERENCE_POLICY_CMD)

    # Move saved policy to input_models directory to serve as the 'fixed' policy
    os.system(MOVE_REFERENCE_POLICY_CMD.format(i=2))

    # Generate mixed policy (which will be very close to the init policy)
    os.system(GENERATE_MIXED_POLICY_CMD)

    # Move mixed policy to input_models directory to serve as the 'fixed' policy
    os.system(MOVE_MIXED_POLICY_CMD.format(i=2))

    # Now, run TRPO_ANALYSIS using as reference the init policy and the mixed policy
    os.system(RUN_TRPO_ANALYSIS_ON_INIT_MIXED_POLICIES.format(exp_idx=i))

    # Now, run TEBPO_MC_ANALYSIS using as refrence the init policy and the mixed policy
    os.system(RUN_TEBPO_MC_ANALYSIS_ON_INIT_MIXED_POLICIES.format(exp_idx=i))

    # Evaluate the two policies
    os.system(RUN_EVALUATION.format(model_idx=1, exp_idx=i))
    os.system(RUN_EVALUATION.format(model_idx=2, exp_idx=i))
