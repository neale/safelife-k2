import numpy as np

import torch
import math
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.



    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())


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




class SafeLifeDistNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """

    def __init__(
            self,
            input_shape,
            out_dim: int,  # Number Actions
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super().__init__()

        if torch.cuda.is_available():
            self.support = support.cuda()
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        # self.advantages = nn.Sequential(
        #     nn.Linear(num_features, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_actions)
        # )
        #
        # self.value_func = nn.Sequential(
        #     nn.Linear(num_features, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

        self.advantage_hidden_layer = nn.Linear(num_features, 256)
        self.advantage_layer = nn.Linear(256, out_dim * atom_size)

        self.value_hidden_layer = nn.Linear(num_features, 256)
        self.value_layer = nn.Linear(256, atom_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q


    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = x.transpose(-1, -3)
        feature = self.cnn(x).flatten(start_dim=1)  # self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist


class SafeLifeQNoise(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        # self.advantages = nn.Sequential(
        #     nn.Linear(num_features, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_actions)
        # )
        #
        # self.value_func = nn.Sequential(
        #     nn.Linear(num_features, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

        # set advantage layer - Noisy
        # self.advantage_noisy = nn.Sequential(NoisyLinear(num_features, 256),
        #                                      nn.ReLU(),
        #                                      NoisyLinear(256, num_actions))
        self.advantage_hidden_layer = NoisyLinear(num_features, 256)
        self.advantage_layer = NoisyLinear(256, num_actions)

        # set value layer
        # self.value_noisy = nn.Sequential(NoisyLinear(num_features, 256),
        #                                      nn.ReLU(),
        #                                      NoisyLinear(256, num_actions))

        self.value_hidden_layer = NoisyLinear(num_features, 256)
        self.value_layer = NoisyLinear(256, num_actions)

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)

        advantage_hidden = F.relu(self.advantage_hidden_layer(x))
        advantages = self.advantage_layer(advantage_hidden)

        value_hidden = F.relu(self.value_hidden_layer(x))
        value = self.value_layer(value_hidden)

        qval = value + advantages - advantages.mean()
        return qval

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


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
