import argparse

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, dqn

from safelife import gym_env
from safelife import file_finder
from training import wrappers
import tensorflow as tf
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, help='SafeLife environment', default='append-still')
parser.add_argument('--epochs', type=int, help='Algorithm Iterations', default=300)
parser.add_argument('--algorithm', default="ppo", choices=['ppo', 'dqn'])
args = parser.parse_args()
logdir_tf = "ray-test-tf"


def env_factory(env_config):
    # For some reason gym.make isn't working.
    # Maybe something to do with the registry not existing for all workers?
    # env = gym.make('safelife-%s-v1' % args.env)
    env = gym_env.SafeLifeEnv(file_finder.safelife_loader('random/' + args.level))
    env = wrappers.BasicSafeLifeWrapper(env)
    env = wrappers.ContEnv(env)
    env = wrappers.MinPerfScheduler(env)
    env = wrappers.RecordingSafeLifeWrapper(
        env, video_name="ray-test-videos/training"+args.algorithm+"-{episode_num}",
        video_recording_freq=1, tf_logger=tf.summary.FileWriter(logdir_tf, tf.get_default_graph())
    )
    return env


ray.init()
register_env("safelife-env", env_factory)

if args.algorithm == "ppo":
    trainer = ppo.PPOTrainer(env="safelife-env", config={
        # ...
    })
else:
    trainer = dqn.DQNTrainer(env="safelife-env", config={
        "lr": 3e-4,
    })


for _ in range(args.epochs):
    print(trainer.train(), '\n')
