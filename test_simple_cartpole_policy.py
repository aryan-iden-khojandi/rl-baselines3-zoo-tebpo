import gym
import matplotlib.pyplot as plt
# from tensorflow import keras
import numpy as np
# import tensorflow as tf


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img


env = gym.make('CartPole-v1')
observation = env.reset()
print(observation)
# plot_environment(env)
# plt.show()


def explicit_policy(obs):

    x, x_dot, theta, theta_dot = obs

    if abs(theta) < 0.03:
        desirable_action = 0 if theta_dot < 0 else 1
    else:
        desirable_action = 0 if theta < 0 else 1

    eps = 0.1
    return (desirable_action + np.random.uniform(-1.0 * eps, eps) - 0.5) > 0

# test the algorithm 500 times for at most 200 steps
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(500):
        action = explicit_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
