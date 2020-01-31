import os
import shutil
import logging.config
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from safelife_factory import environment_factory


USE_CUDA = torch.cuda.is_available()

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '{levelname:8s} {message}',
            'style': '{',
        },
        'dated': {
            'format': '{asctime} {levelname} ({filename}:{lineno}) {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'stream': 'ext://sys.stdout',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'safelife': {
            'level': 'INFO',
            'propagate': False,
            'handlers': ['console'],
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console'],
    }
})


class Timer(object):
    # Just for debugging to see where things are going slowly.
    def __init__(self):
        self.t0 = time.time()

    def msg(self, msg, reset=True):
        print(f"(t={time.time()-self.t0:3g}) {msg}")
        if reset:
            self.t0 = time.time()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = np.random.choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        # input_shape should be (height, width, channels)
        # num_outputs is just the number of actions

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[-1], 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, *input_shape).transpose(-1,-3)
        feature_size = self.features(dummy_input).flatten().size(0)

        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.transpose(-1,-3)  # switch channel to come before height/width
        x = self.features(x)
        x = x.flatten(start_dim=1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if np.random.random() > epsilon:
            state = torch.Tensor(state[np.newaxis])
            q_value = self.forward(state)[0]
            max_q, action = q_value.max(0)
            action = action.item()
        else:
            action = np.random.randint(env.action_space.n)
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(batch_size, replay_buffer, model, target, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = target(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value, next_action = next_q_values.max(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, q_values, next_q_values


def epsilon_schedule(t):
    t0, t1 = (50_000, 1_000_000)
    eps0, eps1 = (1.0, 0.1)
    t = np.clip((t-t0)/(t1-t0), 0, 1)
    return eps0 + (eps1 - eps0) * t


logdir = "./safelife-data/tmp2"
if os.path.exists(logdir):
    shutil.rmtree(logdir)

env = environment_factory(safelife_levels=['random/prune-still'], logdir=logdir)

current_model = DuelingCnnDQN(env.observation_space.shape, env.action_space.n)
target_model = DuelingCnnDQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model = target_model.cuda()

optimizer = optim.Adam(current_model.parameters(), lr=3e-4)

num_frames = int(10e7)
batch_size = 32
gamma = 0.97
replay_initial = 10_000
replay_buffer = ReplayBuffer(100_000)
target_update_freq = 10_000
optimize_freq = 5
record_freq = 25

update_target(current_model, target_model)

state = env.reset()

for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_schedule(frame_idx)
    action = current_model.act(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state

    if done:
        state = env.reset()

    if len(replay_buffer) > replay_initial and frame_idx % optimize_freq == 0:
        loss, q_model, q_target = compute_td_loss(
            batch_size, replay_buffer, current_model, target_model, optimizer)

        if frame_idx % record_freq == 0:
            env.tb_logger.add_scalar("loss", loss.item(), frame_idx)
            env.tb_logger.add_scalar("epsilon", epsilon, frame_idx)
            env.tb_logger.add_scalar(
                "qvals/model_mean", q_model.mean().item(), frame_idx)
            env.tb_logger.add_scalar(
                "qvals/model_max", q_model.max(1)[0].mean().item(), frame_idx)
            env.tb_logger.add_scalar(
                "qvals/target_mean", q_target.mean().item(), frame_idx)
            env.tb_logger.add_scalar(
                "qvals/target_max", q_target.max(1)[0].mean().item(), frame_idx)
            env.tb_logger.flush()

    if frame_idx % target_update_freq == 0:
        update_target(current_model, target_model)
