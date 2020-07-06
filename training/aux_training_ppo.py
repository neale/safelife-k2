import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from .utils import named_output, round_up, LinearSchedule
from . import checkpointing
from .cb_vae import train_encoder, load_state_encoder, encode_state

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class PPO_AUX(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    steps_per_env = 20
    num_minibatches = 4
    epochs_per_batch = 3

    gamma = 0.97
    lmda = 0.95
    learning_rate_aux = 3e-4
    entropy_aux = 0.1

    entropy_clip = 1.0  # don't start regularization until it drops below this
    vf_coef = 0.5
    max_gradient_norm = 5.0
    eps_policy = 0.2  # PPO clipping for policy loss
    eps_value = 0.2  # PPO clipping for value loss
    rescale_policy_eps = False
    min_eps_rescale = 1e-3  # Only relevant if rescale_policy_eps = True
    reward_clip = 0.0
    policy_rectifier = 'relu'  # or 'elu' or ...more to come

    checkpoint_freq = 100000
    num_checkpoints = 3
    report_freq = 960
    test_freq = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None
    epsilon = 0.0  # for exploration


    def __init__(
            self,
            model_aux,
            env_type,
            z_dim,
            buf_size,
            vae_epochs,
            random_projection,
            **kwargs):

        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model_aux = model_aux.to(self.compute_device)
        self.optimizer_aux = optim.Adam(self.model_aux.parameters(), lr=self.learning_rate_aux)
        checkpointing.load_checkpoint(self.logdir, self)
        self.exp = env_type
        
        self.z_dim = z_dim
        self.state_encoder = None
        self.n_random_reward_fns = 1
        self.is_random_projection = random_projection
        self.random_buffer_size = buf_size
        self.train_encoder_epochs = vae_epochs

        if not self.is_random_projection:
            self.load_state_encoder = False
            self.state_encoder_path = 'models/{}/model_save_epoch_100.pt'.format(env_type)
            self.train_state_encoder(envs=self.training_envs)
        self.register_random_reward_functions()
        
    def tensor(self, data, dtype):
        data = np.asanyarray(data)
        return torch.as_tensor(data, device=self.compute_device, dtype=dtype)

    """ 
    generates random linear functions over the encoder space
    the number of functions is given by self.n_random_reward_fns
    if z_dim == 1: the linear function is a 1s matrix
    if rand_proj:  the linear functions are drawn over the state space 
    """
    def register_random_reward_functions(self):
        n_fns = self.n_random_reward_fns
        self.random_fns = []
        for i in range(n_fns):
            if self.is_random_projection:
                self.random_projection = torch.ones(1, 90, 90).uniform_(-1, 1).cuda()
                self.random_projection = self.random_projection.unsqueeze(0).repeat(
                        len(self.training_envs), 1, 1, 1)
                self.random_fns.append(self.random_projection)
            else:
                rfn = torch.ones(self.z_dim).to(self.compute_device)
                if self.z_dim > 1:
                    rfn = rfn.uniform_(0, 1).cuda()
                self.random_fns.append(rfn)
        print ('registering {} random reward functions of dim: {}'.format(n_fns, self.z_dim))
        self.random_fns = torch.stack(self.random_fns)
    

    """
    gets R_aux_i for random funciton i
    if z_dim == 1: then the reward is passed unchanged
    if n_rfn > 1: then the contributions are summed
    """
    def get_aux_rewards(self, states):
        states = torch.stack(states)
        if self.is_random_projection:
            states = states.cuda()
            if self.n_random_reward_fns > 1:
                for i in range(self.n_random_reward_fns):
                    random_projection = self.random_fns[i].transpose(2, 3)
                    reward = torch.einsum('abcd, abde -> abce', states, random_projection)
                    rewards.append(reward.view(reward.size(0), -1).sum(1))
                rewards = torch.stack(rewards).sum(0)

            else:
                random_projection = self.random_projection.transpose(2, 3)
                rewards = torch.einsum('abcd, abde -> abce', states, random_projection)
                rewards = rewards.view(rewards.size(0), -1).sum(1)
        else:
            random_reward_fns = self.random_fns
            self.state_encoder.eval()
            states_z = encode_state(self.state_encoder, states, self.compute_device)
            rewards = []
            if self.n_random_reward_fns > 1:
                for i in range(self.n_random_reward_fns):
                    rr = torch.tensor(random_reward_fns.T[:, i]).unsqueeze(1)
                    rewards.append(torch.mm(states_z, rr))
                rewards = torch.stack(rewards)
                rewards = rewards.sum(0)
            else:
                rewards = torch.mm(states_z, random_reward_fns.T)
        return rewards
    

    """
    Takes an environment object and does hard-coded preprocessing
    Specifically: we average_pool the rendered (350, 350, 3) state
    convert to intensity values: (350, 350, 3) --> (350, 350, 1)
    Avg_Pool2d(state, kernel=5x5, stride=4): (350, 350, 1) --> (90, 90, 1)
    Normalize by pixel scale: values/255.
    """
    def preprocess_state(self, env, reset=False, return_original=False):
        if reset:
            _ = env.reset()
        obs = render_board(env.game.board, env.game.goals, env.game.orientation)
        obs = np.asarray(obs)
        obsp = torch.from_numpy(np.matmul(obs[:, :, :3], [0.299, 0.587, 0.114]))
        obsp = obsp.unsqueeze(0) # [1, batch, H, W]
        if obsp.size(-1) == 210: # test env
            obsp = F.avg_pool2d(obsp, 2, 2)/255.
        else: # random env
            obsp = F.avg_pool2d(obsp, 5, 4)/255.
        if return_original:
            ret = (obsp.float(), obs)
        else:
            ret = obsp.float()
        return ret


    """ 
    Trains CB-VAE on preprocessed states
    states are collected through <buffer_size> random actions
    CB-VAE is trained for <n_epochs> 
    """
    def train_state_encoder(self, envs):
        if self.load_state_encoder:
            print ('loading state encoder')
            self.state_encoder = load_state_encoder(z_dim=self.z_dim,
                                                    path=self.state_encoder_path,
                                                    device=self.compute_device)
            return
        envs = [e.unwrapped for e in envs]
        states = [e.last_obs if hasattr(e, 'last_obs') else e.reset() for e in envs]
        print ('gathering data from every env N={}'.format(len(envs)))
        buffer = []
        buffer_size = int(self.random_buffer_size // len(envs))
        for env in envs:
            for _ in range(buffer_size):
                action = np.random.choice(env.action_space.n, 1)[0]
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()
                obs = self.preprocess_state(env, return_original=False)
                buffer.append(obs)
        print ('collected data for state encoder')
        buffer_th = torch.stack(buffer)
        buffer_np = buffer_th.cpu().numpy()
        np.save('buffers/{}/state_buffer'.format(self.exp), buffer_np)
        self.state_encoder = train_encoder(
                device=self.compute_device,
                data=buffer_th,
                z_dim=self.z_dim,
                training_epochs=self.train_encoder_epochs,
                exp=self.exp)
    

    """ 
    Take actions w.r.t R_aux. 
    Train PPO normally, yet replace reward with random reward
    If we use rand_proj, then render states before getting random reward
    otherwise encode them
    """
    @named_output('states actions rewards done policies values')
    def take_one_step_aux(self, envs):
        states = [e.last_obs if hasattr(e, 'last_obs') else e.reset() for e in envs]
        
        rendered_states = [
                e.last_rendered_obs if hasattr(e, 'last_rendered_obs') else self.preprocess_state(e, reset=True)
                for e in envs
                ]

        aux_rewards = self.get_aux_rewards(rendered_states)
        self.aux_rewards = aux_rewards
        aux_rewards = aux_rewards.squeeze(-1).tolist() # [batch, actions]

        tensor_states = self.tensor(states, dtype=torch.float32)
        values_q, policies = self.model_aux(tensor_states)
        values = values_q.mean(1)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()

        actions = []
        rewards = []
        dones = []
        for i, (policy, env) in enumerate(zip(policies, envs)):
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, _, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            env.last_rendered_obs = self.preprocess_state(env)
            actions.append(action)
            dones.append(done)

        rewards = aux_rewards
        return states, actions, rewards, dones, policies, values


    def run_test_envs(self):
        """
        Run each testing environment until completion.

        It's assumed that the testing environments take care of their own
        logging via wrappers.
        """
        test_envs = self.testing_envs or []
        while test_envs:
            data = self.take_one_step_aux(test_envs)
            test_envs = [
                env for env, done in zip(test_envs, data.done) if not done
            ]

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env, flat=True):
        """
        Run each environment a number of steps and calculate advantages.

        Parameters
        ----------
        steps_per_env : int
            Number of steps to take per environment.
        flat : bool
            If True, each output tensor will have shape
            ``(steps_per_env * num_env, ...)``.
            Otherwise, shape will be ``(steps_per_env, num_env, ...)``.
        """
        steps = [self.take_one_step_aux(self.training_envs) for _ in range(steps_per_env)] # [steps]
        final_states = [e.last_obs for e in self.training_envs]
        tensor_states = self.tensor(final_states, dtype=torch.float32)
        final_vals = self.model_aux(tensor_states)[0]
        final_vals = final_vals.mean(1).detach().cpu().numpy()  # adjust for learning Q function
        values = np.array([s.values for s in steps] + [final_vals])

        rewards = np.array([s.rewards for s in steps])
        done = np.array([s.done for s in steps])
        reward_mask = ~done

        # Calculate the discounted rewards
        gamma = self.gamma
        lmda = self.lmda
        returns = rewards.copy()
        # shape of returns is different
        returns[-1] += gamma * final_vals * reward_mask[-1]
        advantages = rewards + gamma * reward_mask * values[1:] - values[:-1]
        for i in range(steps_per_env - 2, -1, -1):
            returns[i] += gamma * reward_mask[i] * returns[i+1]
            advantages[i] += lmda * reward_mask[i] * advantages[i+1]

        # Calculate the probability of taking each selected action
        policies = np.array([s.policies for s in steps])
        actions = np.array([s.actions for s in steps])
        probs = np.take_along_axis(
            policies, actions[..., np.newaxis], axis=-1)[..., 0]

        def t(x, dtype=torch.float32):
            if flat:
                x = np.asanyarray(x)
                x = x.reshape(-1, *x.shape[2:])
            return self.tensor(x, dtype=dtype)

        self.num_steps += actions.size
        self.num_episodes += np.sum(done)
        return (
                t([s.states for s in steps]), t(actions, torch.int64),
                t(probs), t(returns), t(advantages), t(values[:-1]),
                )

    def calculate_loss(
            self, states, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        ne_step = s
        """
        values, policy = self.model_aux(states)
        values = values.mean(1)  # adjust for learning Q function


        a_policy = torch.gather(policy, -1, actions[..., np.newaxis])[..., 0]

        prob_diff = advantages.sign() * (1 - a_policy / old_policy)
        policy_loss = advantages.abs() * torch.clamp(prob_diff, min=-self.eps_policy)
        policy_loss = policy_loss.mean()

        v_clip = old_values + torch.clamp(
            values - old_values, min=-self.eps_value, max=+self.eps_value)
        value_loss = torch.max((v_clip - returns)**2, (values - returns)**2)
        value_loss = value_loss.mean()

        entropy = torch.sum(-policy * torch.log(policy + 1e-12), dim=-1)
        entropy_loss = torch.clamp(entropy.mean(), max=self.entropy_clip)
        entropy_loss *= -self.entropy_aux

        return entropy, policy_loss + value_loss * self.vf_coef + entropy_loss

    def train_batch(self, batch):
        idx = np.arange(len(batch.states))
        for _ in range(self.epochs_per_batch):
            np.random.shuffle(idx)
            for k in idx.reshape(self.num_minibatches, -1):
                entropy, loss = self.calculate_loss(
                    batch.states[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
                self.optimizer_aux.zero_grad()
                loss.backward()
                self.optimizer_aux.step()

    def train(self, steps):
        print ('starting training')
        max_steps = self.num_steps + steps
        
        while self.num_steps < max_steps:
            next_checkpoint = round_up(self.num_steps, self.checkpoint_freq)
            next_report = round_up(self.num_steps, self.report_freq)
            next_test = round_up(self.num_steps, self.test_freq)
            batch = self.gen_training_batch(self.steps_per_env)
            self.train_batch(batch)

            n = self.num_steps
            
            if n >= next_report and self.summary_writer is not None:
                writer = self.summary_writer
                entropy, loss = self.calculate_loss(
                    batch.states, batch.actions, batch.action_prob,
                    batch.values, batch.returns, batch.advantages)
                loss = loss.item()
                entropy = entropy.mean().item()
                values = batch.values.mean().item()
                advantages = batch.advantages.mean().item()
                
                aux_reward = self.aux_rewards.mean().item()

                logger.info(
                    "n=%i: loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g, aux_r=%0.3f",
                    n, loss, entropy, values, advantages, aux_reward)
                writer.add_scalar("training/loss", loss, n)
                writer.add_scalar("training/entropy", entropy, n)
                writer.add_scalar("training/values", values, n)
                writer.add_scalar("training/advantages", advantages, n)
                writer.add_scalar("training/aux_reward", aux_reward, n)
                writer.flush()

            if n >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model_aux', 'optimizer_aux',
                ], 'aux_')

            if n >= next_test:
                self.run_test_envs()

        # Closing up
        for env in self.training_envs:
            env.close()
        for env in self.testing_envs:
            env.close()
