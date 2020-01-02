import os
import sys
import tensorflow as tf
from safelife.safelife_env import SafeLifeEnv; SafeLifeEnv.register()
import safelife.env_wrappers as env_wrappers
import gym
from gym import Wrapper

# ##Create the safelife env with wrappers
# env = SafeLifeEnv(view_shape=(25,25), level_iterator=('append-still'))
# env = env_wrappers.MovementBonusWrapper(env)
# # env = env_wrappers.ExtraExitBonus(env)
# env.register()

class ExtraExitBonus(Wrapper):
    bonus = 0.5

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and not info['times_up']:
            reward += self.scheduled(self.bonus) * self.episode_reward

        return obs, reward, done, info

# env = SafeLifeEnv(view_shape=(15,15), level_iterator=('append-still'))
# env = env_wrappers.MovementBonusWrapper(env)
# env = ExtraExitBonus(env)
# env.register()#id='safelife-custom-v1')

def env_factory():
    env = SafeLifeEnv(view_shape=(25,25), level_iterator=('append-still'))
    env = env_wrappers.MovementBonusWrapper(env)
    env = ExtraExitBonus(env)
    # env = gym.make(env)
    return env

gym.register(id='safelife-custom-v1', entry_point=env_factory,)

# env2 = gym.make('safelife-custom-v1')

# from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.layers import Conv2d, Dense, BatchnormActivationDropout

# Resetting tensorflow graph as the network has changed.
tf.reset_default_graph()

#Graph Scheduling
schedule_params = SimpleSchedule()

##Agent
agent_params = DQNAgentParameters()
agent_params.input_filter = NoInputFilter()
agent_params.output_filter = NoOutputFilter()

agent_params.network_wrappers['main'].input_embedders_parameters = {
            'observation': InputEmbedderParameters(
                scheme=[
                    Conv2d(32, 5, 2),
                    BatchnormActivationDropout(batchnorm=False, activation_function='relu'),
                    Conv2d(64, 3, 1),
                    BatchnormActivationDropout(batchnorm=False, activation_function='relu'),
                    Conv2d(64, 3, 1),
                    BatchnormActivationDropout(batchnorm=False, activation_function='relu'),
                    Dense(512),
                    BatchnormActivationDropout(activation_function='relu'),
                    Dense(512),
                    BatchnormActivationDropout(activation_function='relu'),
                ],
                activation_function='none')
            }

# DQN Specific Parameters

# ER Size
# agent_params.memory.max_size = (MemoryGranularity.Transitions, 40000)


# define the environment parameters
env_params = GymVectorEnvironment(level='safelife-prune-still-v1')
# env_params = GymVectorEnvironment(level=env_factory)
# env_params.additional_simulator_parameters = {'bit_length': bit_length, 'mean_zero': True}

# Clipped PPO
# agent_params = ClippedPPOAgentParameters()
# agent_params.network_wrappers['main'].input_embedders_parameters = {
#     'state': InputEmbedderParameters(scheme=[]),
#     'desired_goal': InputEmbedderParameters(scheme=[])
# }

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=SimpleSchedule()
)

graph_manager.improve()