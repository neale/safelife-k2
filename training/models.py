import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training.layers import NoisyLinear

USE_CUDA = torch.cuda.is_available()

def safelife_cnn(input_shape):
    """
    Defines a CNN with good default values for safelife.

    This works best for inputs of size 25x25.

    Parameters
    ----------
    input_shape : tuple of ints
        Height, width, and number of channels for the board.

    Returns
    -------
    cnn : torch.nn.Sequential
    output_shape : tuple of ints
        Channels, width, and height.

    Returns both the CNN module and the final output shape.
    """
    h, w, c = input_shape
    cnn = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )
    h = (h-4+1)//2
    h = (h-2+1)//2
    h = (h-2)
    w = (w-4+1)//2
    w = (w-2+1)//2
    w = (w-2)
    return cnn, (64, w, h)


class SafeLifeQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.advantages = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        value = self.value_func(x)
        qval = value + advantages - advantages.mean()
        return qval


class SafeLifePolicyNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.dense = nn.Sequential([
            nn.Linear(num_features, 512),
            nn.ReLU(),
        ])
        self.logits = nn.Linear(512, num_actions)
        self.value_func = nn.Linear(512, 1)

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        x = self.dense(x)
        value = self.value_func(x)[...,0]
        advantages = F.softmax(self.logits(x), dim=-1)
        return value, advantages


class RainbowDQNNet(nn.Module):
    def __init__(self, input_shape, num_atoms, Vmin, Vmax):
        super(RainbowDQNNet, self).__init__()

        self.input_shape = input_shape
        self.num_actions = 9
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax


        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)

        self.noisy_value1 = NoisyLinear(num_features, 256, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(256, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(num_features, 256, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(256, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, obs):
        batch_size = obs.size(0)
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    # def feature_size(self):
    #     return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    # def act(self, state):
    #     state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
    #     dist = self.forward(state).data.cpu()
    #     dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
    #     action = dist.sum(2).max(1)[1].numpy()[0]
    #     return action
