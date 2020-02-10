import os
import sys
import tensorflow as tf
from safelife.safelife_env import SafeLifeEnv; SafeLifeEnv.register()
import safelife.env_wrappers as env_wrappers
import gym
from gym import Wrapper
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter


# ##Create the safelife env with wrappers
# env = SafeLifeEnv(view_shape=(25,25), level_iterator=('append-still'))
# env = env_wrappers.MovementBonusWrapper(env)
# # env = env_wrappers.ExtraExitBonus(env)
# env.register()

# Parameters
class SafelifeParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()
        # self.level_iterator = 'prune-still-v1'

    @property
    def path(self):
        return 'safelife.safelife_env'

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

# def env_factory():
#     env = SafeLifeEnv(view_shape=(25,25), level_iterator=('prune-still'))
#     # env = env_wrappers.MovementBonusWrapper(env)
#     # env = ExtraExitBonus(env)
#     # env = gym.make(env)
#     return env

# gym.register(id='safelife-custom-v1', entry_point=env_factory,)

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
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.base_parameters import TaskParameters
# from rl_coach.environments.environment import SingleLevelSelection

# Resetting tensorflow graph as the network has changed.
# tf.reset_default_graph()

#Graph Scheduling
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(1000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

###Viz
vis_params = VisualizationParameters()
# vis_params.level_iterator = 'prune-still-v1'


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
# env_params = SafelifeParameters(level='prune-still-v1')
# env_params = SafeLifeEnv(level_iterator='prune-still-v1')
# env_params = GymVectorEnvironment(level='safelife-prune-still-v1')
# env_params = GymVectorEnvironment(level=env_factory)
# env_params.additional_simulator_parameters = {'bit_length': bit_length, 'mean_zero': True}

env_params = GymVectorEnvironment(level=os.path.join(
    os.path.dirname(__file__),
    'safelife_factory.py:environment_factory'
))

# env_params = GymVectorEnvironment(level=
#     'safelife_factory.py:environment_factory'
# )


####Simple Graph
# graph_manager = BasicRLGraphManager(
#     agent_params=agent_params,
#     env_params=env_params,
#     schedule_params=SimpleSchedule()
# )

task1 = TaskParameters()
task1.experiment_path = os.path.join(os.path.dirname(__file__), 'exp1')

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)


graph_manager.create_graph(task1)
graph_manager.improve()