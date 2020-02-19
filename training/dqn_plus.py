import os
import glob

import numpy as np

import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.safelife_env import SafeLifeEnv
from training.rainbow_replay import ReplayBuffer, PrioritizedReplayBuffer


USE_CUDA = torch.cuda.is_available()


def round_up(x, r):
    return x + r - x % r


# class ReplayBuffer(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.idx = 0
#         self.buffer = np.zeros(capacity, dtype=object)
#
#     def push(self, *data):
#         self.buffer[self.idx % self.capacity] = data
#         self.idx += 1
#
#     def sample(self, batch_size):
#         sub_buffer = self.buffer[:self.idx]
#         data = np.random.choice(sub_buffer, batch_size, replace=False)
#         return zip(*data)
#
#     def __len__(self):
#         return min(self.idx, self.capacity)


class DQN(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    gamma = 0.97
    # beta = 0.4
    prior_eps = 1e-5
    training_batch_size = 64
    optimize_freq = 16
    learning_rate = 3e-4

    replay_initial = 40000
    replay_size = 100000
    target_update_freq = 10000

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 256
    test_freq = 100000

    v_min = - 10.0
    v_max = 10.0
    atom_size = 51

    priority = False
    noise = False
    distributed = True

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    support = torch.linspace(v_min, v_max, atom_size).to(compute_device)

    training_envs = None
    testing_envs = None

    def __init__(self, training_model, target_model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.training_model = training_model.to(self.compute_device)
        self.target_model = target_model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.training_model.parameters(), lr=self.learning_rate)

        obs_dim = self.training_envs[0].observation_space.shape
        #         print(obs_dim)
        action_dim = self.training_envs[0].action_space.n


        if self.priority:
            self.replay_buffer = PrioritizedReplayBuffer(obs_dim, size = self.replay_size, batch_size = self.training_batch_size)

        else:
            self.replay_buffer = ReplayBuffer(obs_dim, size=self.replay_size, batch_size=self.training_batch_size)

        self.load_checkpoint()

    @property
    def epsilon(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 1e6
        y1 = 1.0
        y2 = 0.1
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2-y1) * np.clip(t, 0, 1)

    @property
    def beta(self):
        # hardcode this for now
        t1 = 1e5
        t2 = 3e6
        y1 = 0.4
        y2 = 1.0
        t = (self.num_steps - t1) / (t2 - t1)
        return y1 + (y2 - y1) * np.clip(t, 0, 1)

    def get_next_checkpoint(self):
        if self.logdir is None:
            return None
        num_steps = self.num_steps
        return os.path.join(self.logdir, 'checkpoint-%i.data' % num_steps)

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.get_next_checkpoint()
        torch.save({
            'num_steps': self.num_steps,
            'num_episodes': self.num_episodes,
            'training_model_state_dict': self.training_model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        for old_checkpoint in self.get_all_checkpoints()[:-self.num_checkpoints]:
            os.remove(old_checkpoint)

    def get_all_checkpoints(self):
        if self.logdir is None:
            return []
        files = glob.glob(os.path.join(self.logdir, 'checkpoint-*.data'))

        def step_from_checkpoint(f):
            try:
                return int(os.path.basename(f)[11:-5])
            except ValueError:
                return -1

        files = [f for f in files if step_from_checkpoint(f) >= 0]
        return sorted(files, key=step_from_checkpoint)

    def load_checkpoint(self, path=None):
        if path is None:
            checkpoints = self.get_all_checkpoints()
            path = checkpoints and checkpoints[-1]
        if not path or not os.path.exists(path):
            return
        checkpoint = torch.load(path)
        self.training_model.load_state_dict(checkpoint['training_model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_steps = checkpoint['num_steps']
        self.num_episodes = checkpoint['num_episodes']

        # very, very small amount of safelife specific code:
        SafeLifeEnv.global_counter.num_steps = self.num_steps
        SafeLifeEnv.global_counter.episodes_started = self.num_episodes
        SafeLifeEnv.global_counter.episodes_completed = self.num_episodes

    def update_target(self):
        self.target_model.load_state_dict(self.training_model.state_dict())

    def run_test_envs(self):
        # Just run one episode of each test environment.
        # Assumes that the environments themselves handle logging.
        for env in self.testing_envs:
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor([state], device=self.compute_device, dtype=torch.float32)
                qvals = self.training_model(state).detach().cpu().numpy().ravel()

                num_actions = qvals.shape
                action = np.argmax(qvals)
                random_action = np.random.randint(num_actions, size=1)
                use_random = np.random.random(1) < self.epsilon
                action = np.choose(use_random, [action, random_action])

                state, reward, done, info = env.step(int(action))

    def collect_data(self):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in self.training_envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        qvals = self.training_model(tensor_states).detach().cpu().numpy()

        num_states, num_actions = qvals.shape
        actions = np.argmax(qvals, axis=-1)
        random_actions = np.random.randint(num_actions, size=num_states)
        use_random = np.random.random(num_states) < self.epsilon
        actions = np.choose(use_random, [actions, random_actions])

        for env, state, action in zip(self.training_envs, states, actions):
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
                self.num_episodes += 1
            env.last_state = next_state
            self.replay_buffer.store(state, action, reward, next_state, done)

        self.num_steps += len(states)

    def optimize(self, report=False):
        if len(self.replay_buffer) < self.replay_initial:
            return

        if self.priority:
            samples = self.replay_buffer.sample_batch(self.beta)
        else:
            samples = self.replay_buffer.sample_batch()


        device = self.compute_device  # for shortening the following lines

        if not self.distributed:
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.LongTensor(samples["acts"]).to(device)
            reward = torch.FloatTensor(samples["rews"]).to(device)
            done = torch.FloatTensor(samples["done"]).to(device)
            if self.priority:
                weights = torch.FloatTensor(samples["weights"]).to(device)
                indices = samples["indices"]


            q_values = self.training_model(state)
            next_q_values = self.target_model(next_state).detach()

            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value, next_action = next_q_values.max(1)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            elem_loss = (q_value - expected_q_value)**2 #torch.mean((q_value - expected_q_value)**2)
            if self.priority:
                loss_for_prior = elem_loss * weights
                loss = torch.mean(loss_for_prior)
            else:
                loss = torch.mean(elem_loss)

        else:
            elementwise_loss = self._compute_dqn_loss(samples)
            if self.priority:
                weights = torch.FloatTensor(samples["weights"]).to(device)
                indices = samples["indices"]
                loss_for_prior = elementwise_loss * weights
                loss = torch.mean(loss_for_prior)
            else:
                loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.priority:
        ###Update Priorities
            priors = loss_for_prior.detach().cpu().numpy()
            new_priorities = priors + self.prior_eps
            self.replay_buffer.update_priorities(indices, new_priorities)

        if self.noise:
            # NoisyNet: reset noise
            self.training_model.reset_noise()
            self.target_model.reset_noise()

        writer = self.summary_writer
        n = self.num_steps
        if report and self.summary_writer is not None:
            writer.add_scalar("loss", loss.item(), n)
            writer.add_scalar("epsilon", self.epsilon, n)
            if not self.distributed:
                writer.add_scalar("qvals/model_mean", q_values.mean().item(), n)
                writer.add_scalar("qvals/model_max", q_values.max(1)[0].mean().item(), n)
                writer.add_scalar("qvals/target_mean", next_q_values.mean().item(), n)
                writer.add_scalar("qvals/target_max", next_q_value.mean().item(), n)
            writer.flush()

    def train(self, steps):
        needs_report = True

        for _ in range(int(steps)):
            num_steps = self.num_steps
            next_opt = round_up(num_steps, self.optimize_freq)
            next_update = round_up(num_steps, self.target_update_freq)
            next_checkpoint = round_up(num_steps, self.checkpoint_freq)
            next_report = round_up(num_steps, self.report_freq)
            next_test = round_up(num_steps, self.test_freq)

            self.collect_data()

            num_steps = self.num_steps

            if len(self.replay_buffer) < self.replay_initial:
                continue

            if num_steps >= next_report:
                needs_report = True

            if num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            if num_steps >= next_update:
                self.target_model.load_state_dict(self.training_model.state_dict())

            if num_steps >= next_checkpoint:
                self.save_checkpoint()

            if num_steps >= next_test:
                self.run_test_envs()

    def _compute_dqn_loss(self, samples) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.compute_device  # for shortening the following lines
        batch_size = self.training_batch_size

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.training_model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.atom_size, batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(batch_size, self.atom_size)
                    .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.training_model.dist(state)
        log_p = torch.log(dist[range(batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
