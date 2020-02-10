import gym, ray
from ray.rllib.agents import dqn
import numpy as np
import argparse
ray.init()

from safelife.ray_env import SafeLifeRay

parser = argparse.ArgumentParser()

parser.add_argument('--level', type=str, help='Level Iterator', default='random/append-still')
parser.add_argument('--epochs', type=int, help='Algorithm Iterations', default=200)

env_level_it = vars(parser.parse_args())['level']
epochs = vars(parser.parse_args())['epochs']


trainer = dqn.DQNTrainer(env=SafeLifeRay, config={
    "env_config": {'level_iterator' : env_level_it},
    "lr" : 3e-4,

})

for _ in range(epochs):
    print(trainer.train())


# from ray import tune
# from ray.rllib.agents.ppo import DQNTrainer
#
# tune.run(DQNTrainer, config={"env": "SafeLifeEnv"})