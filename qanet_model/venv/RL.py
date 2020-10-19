from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import spaces
import numpy as np
#import test_exercises

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

ray.init(ignore_reinit_error=True, log_to_driver=False)

action_space_map = {
    "discrete_10": spaces.Discrete(10),
    "box_1": spaces.Box(0, 1, shape=(1,)),
    "box_3x1": spaces.Box(-2, 2, shape=(3, 1)),
    "multi_discrete": spaces.MultiDiscrete([5, 2, 2, 4])
}

action_space_jumble = {
    "discrete_10": 1,
    "box_1": np.array([0.89089584]),
    "box_3x1": np.array([[-1.2657754], [-1.6528835], [0.5982418]]),
    "multi_discrete": np.array([0, 0, 0, 2]),
}

for space_id, state in action_space_jumble.items():
    assert action_space_map[space_id].contains(state), (
        "Looks like {} to {} is matched incorrectly.".format(space_id, state))

print("Success!")

import gym
from gym import spaces
import numpy as np
#import test_exercises


class MyEnv(gym.Env):
    def __init__(self, env_config=None, T=30):
        env_config = env_config or {}
        self.state = np.random.normal(0, 1, 2)  # Start at beginning of the chain
        self._horizon = env_config.get("T", T)
        self._counter = 0  # For terminating the episode
        self._setup_spaces()

    def _setup_spaces(self):
        ##############
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-10, 10, [2, ])
        ##############

    def step(self, action):
        self.state[0] = 3 / 4 * (2 * action - 1) * self.state[0] + np.random.normal(0, 0.5, 1)
        self.state[1] = 3 / 4 * (1 - 2 * action) * self.state[1] + np.random.normal(0, 0.5, 1)
        reward = 2 * self.state[0] + self.state[1] - 1 / 4 * (2 * action - 1)

        self._counter += 1
        done = self._counter >= self._horizon
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.normal(0, 1, 2)
        self._counter = 0
        return self.state

trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 64
trainer_config["sgd_minibatch_size"] = 64
trainer_config["num_sgd_iter"] = 10


trainer = PPOTrainer(trainer_config, MyEnv);
for i in range(50):
    print("Training iteration {}...".format(i))
    trainer.train()

cumulative_reward_list = []
M = 100
for rep in range(M):
    env = MyEnv({})
    state = env.reset()

    done = False
    cumulative_reward = 0
    while not done:
        action = trainer.compute_action(state)
        #print(action, state)
        state, reward, done, results = env.step(action)
        cumulative_reward += reward
    cumulative_reward_list.append(cumulative_reward)
    #print("Cumulative reward you've received is: {}".format(np.mean(cumulative_reward_list)))

print("Cumulative reward you've received is: {}".format(np.mean(cumulative_reward_list)))