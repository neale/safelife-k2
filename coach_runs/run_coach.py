import os
import tensorflow as tf

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

# Graph Scheduling
schedule_params = SimpleSchedule()

# Agent
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
env_params = GymVectorEnvironment(level=os.path.join(
    os.path.dirname(__file__),
    'safelife_factory.py:environment_factory'
))
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
