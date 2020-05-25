import logging
import numpy as np

import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from .utils import named_output, round_up, LinearSchedule
from . import checkpointing
from .aae import *
from .cb_vae import train_encoder, load_state_encoder, encode_state

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class PPO(object):
    summary_writer = None
    logdir = None

    num_steps = 0
    num_episodes = 0

    steps_per_env = 20
    num_minibatches = 4
    epochs_per_batch = 3

    gamma = 0.97
    lmda = 0.95
    learning_rate = 3e-4
    learning_rate_aup = 3e-4
    entropy_reg = 0.01 
    entropy_aup = 0.1

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


    def __init__(self, ppo_agent, model, model_aup, env_type, z_dim, agent_index, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model = model.to(self.compute_device)
        self.model_aup = model_aup.to(self.compute_device)
        self.idx = agent_index
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.optimizer_aup = optim.Adam(
            self.model_aup.parameters(), lr=self.learning_rate_aup)
        checkpointing.load_checkpoint(self.logdir, self)
        self.exp = env_type
        """ AUP-specific parameters """
        self.z_dim = z_dim
        self.state_encoder = None
        self.n_random_reward_fns = 1
        self.training_aup = True    # indicator: currently training random reward agent
        self.impact_training = True # indicator: use penalty (True) or not (False) in take_one_step
        self.switch_to_ppo = False  # indicator: turn on PPO (True) or off (False)
        self.train_aup_steps=200e3
        self.use_scale = False
        self.value_agent = ppo_agent
        if self.impact_training:
            self.lamb_schedule = LinearSchedule(1.8e6, initial_p=3, final_p=3)
            self.random_proj = False
            if not self.random_proj:
                self.load_rendered_state_buffer = False
                self.state_encoder_path = None#'models/{}/100/model_save_epoch_100.pt'.format(self.exp)
                self.train_state_encoder(envs=self.training_envs)
                self.register_random_reward_functions()
            else:
                self.random_projection = torch.ones(1, 90, 90).uniform_(-1, 1).cuda()
                self.random_projection = self.random_projection.unsqueeze(0).repeat(8, 1, 1, 1)
        
    
    def register_random_reward_functions(self):
        """ generates random linear funcitons over the encoder space """
        n_fns = self.n_random_reward_fns
        self.random_fns = []
        for i in range(n_fns):
            rfn = torch.ones(self.z_dim).to(self.compute_device)
            # rfn = rfn.uniform_(0, 1).cuda()
            plt.plot(rfn.cpu().numpy())
            plt.savefig('random_reward_{}.png'.format(i))
            self.random_fns.append(rfn)
        print ('registering random reward function of dim ', self.z_dim)
        self.random_fns = torch.stack(self.random_fns)

    def get_rand_rewards(self, states):
        random_reward_fns = self.random_fns
        states = torch.stack(states)
        if self.random_proj:
            states = states.cuda()
            random_projection = self.random_projection.transpose(2, 3)
            rewards = torch.einsum('abcd, abde -> abce', states, random_projection)
            rewards = rewards.view(rewards.size(0), -1).sum(1)
        else:
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
    
    def batch_preprocess_states(self, states, envs):
        obs = [render_board(env.game.board, env.game.goals, env.game.orientation) for env in envs]
        obs = np.asarray(obs)
        obs = torch.from_numpy(np.matmul(obs[:, :, :, :3], [0.299, 0.587, 0.114]))
        obs = obs.unsqueeze(0) # [1, batch, H, W]
        if obs.size(-1) == 210: # test env
            obs = F.avg_pool2d(obs, 2, 2)/255.
        else: # random env
            obs = F.avg_pool2d(obs, 5, 4)/255.
        return obs.float()
    
    def render_state(self, env, reset=False, return_original=False):
        if reset:
            _ = env.reset
        obs = render_board(env.game.board, env.game.goals, env.game.orientation)
        obs = np.asarray(obs)
        obsp = torch.from_numpy(obs)
        obsp = obsp.float()/255.
        return obsp

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

    def train_state_encoder(self, envs, buffer_size=100e3):
        if self.state_encoder_path is not None:
            print ('loading state encoder')
            self.state_encoder = load_state_encoder(z_dim=self.z_dim,
                                                    path=self.state_encoder_path,
                                                    device=self.compute_device)
            return
        if self.load_rendered_state_buffer:
            buffer_np = np.load('buffers/{}/state_buffer.npy'.format(self.exp))
            buffer_th = torch.from_numpy(buffer_np).float()
        else:
            envs = [e.unwrapped for e in envs]
            states = [
                e.last_obs if hasattr(e, 'last_obs') else e.reset() for e in envs
            ]
            print ('gathering data from every env N={}'.format(len(envs)))
            buffer = []
            samples = []
            buffer_size = int(buffer_size // len(envs))
            for env in envs:
                for _ in range(buffer_size):
                    action = np.random.choice(env.action_space.n, 1)[0]
                    obs, _, done, _ = env.step(action)
                    if done:
                        obs = env.reset()
                    obs = self.preprocess_state(env, return_original=False)
                    buffer.append(obs)
                    #samples.append(sample)
            print ('collected data')
            buffer_th = torch.stack(buffer)
            buffer_np = buffer_th.cpu().numpy()
            #np.random.shuffle(samples)
            #np.save('buffers/{}/example_data'.format(self.exp), samples[:200])
            np.save('buffers/{}/state_buffer'.format(self.exp), buffer_np)
        
        del samples
        self.state_encoder = train_encoder(device=self.compute_device,
                data=buffer_th,
                z_dim=self.z_dim,
                training_epochs=100,
                exp=self.exp,
                )
        return

    @named_output('states actions rewards done policies values')
    def take_one_step(self, envs):
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset()
            for e in envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values_q, policies = self.model(tensor_states)
        values = values_q.mean(1)
        values_q_aup, policies_aup = self.model_aup(tensor_states)
        
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()
        
        actions = []
        rewards = []
        dones = []
        for i, (policy, env) in enumerate(zip(policies, envs)):
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)

            if self.impact_training:
                # calculate AUP penalty
                noop_value = values_q_aup[i, 0]
                max_value = values_q_aup[i].max()
                penalty = (max_value - noop_value).abs()
                self.scale = noop_value
                if self.use_scale:
                    scale = noop_value
                else:
                    scale = 1.
                lamb = self.lamb_schedule.value(self.num_steps-self.train_aup_steps)
                self.lamb = lamb
                self.penalty = penalty
                reward = reward - lamb * (penalty / scale)
                reward = reward.cpu().tolist()
            rewards.append(reward)
            dones.append(done)
       
        # print (rewards)
        return states, actions, rewards, dones, policies, values

    @named_output('states actions rewards done policies values')
    def take_one_step_rand(self, envs):
        states = [
                e.last_obs if hasattr(e, 'last_obs') else e.reset()
                for e in envs
                ]
        if self.random_proj:
            rendered_states = [
                    e.last_rendered_obs if hasattr(e, 'last_rendered_obs') else self.preprocess_state(e, reset=True)
                    for e in envs
                    ]
        else:
            rendered_states = [
                    e.last_rendered_obs if hasattr(e, 'last_rendered_obs') else self.preprocess_state(e, reset=True)
                    for e in envs
                    ]
        random_rewards = self.get_rand_rewards(rendered_states)
        self.rand_rewards = random_rewards  # *shrug* for logging
        random_rewards = random_rewards.squeeze(-1).tolist() # [batch, actions]

        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values_q, policies = self.model_aup(tensor_states)
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
            if self.random_proj:
                env.last_rendered_obs = self.preprocess_state(env)
            else:
                env.last_rendered_obs = self.preprocess_state(env)
            actions.append(action)
            dones.append(done)

        rewards = random_rewards
        return states, actions, rewards, dones, policies, values


    def run_test_envs(self):
        """
        Run each testing environment until completion.

        It's assumed that the testing environments take care of their own
        logging via wrappers.
        """
        test_envs = self.testing_envs or []
        while test_envs:
            data = self.take_one_step(test_envs)
            test_envs = [
                env for env, done in zip(test_envs, data.done) if not done
            ]

    @named_output('states actions action_prob returns advantages values, values2')
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
        if self.training_aup:
            model = self.model_aup
            take_one_step = self.take_one_step_rand 
        else:
            model = self.model
            take_one_step = self.take_one_step

        steps = [take_one_step(self.training_envs) for _ in range(steps_per_env)] # [steps]
        final_states = [e.last_obs for e in self.training_envs]
        tensor_states = torch.tensor(
            final_states, device=self.compute_device, dtype=torch.float32)
        final_vals = model(tensor_states)[0]
        final_vals = final_vals.mean(1).detach().cpu().numpy()  # adjust for learning Q function
        values = np.array([s.values for s in steps] + [final_vals])

        with torch.no_grad():
            if self.value_agent is not None:
                final_vals2 = self.value_agent.model(tensor_states)[0]
                final_vals2 = final_vals2.mean(1).detach().cpu().numpy()  
                values2 = np.array([s.values for s in steps] + [final_vals])
            else:
                values2 = torch.zeros(*values.shape)

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
            return torch.tensor(x, device=self.compute_device, dtype=dtype)

        self.num_steps += actions.size
        self.num_episodes += np.sum(done)
        return (
                t([s.states for s in steps]), t(actions, torch.int64),
                t(probs), t(returns), t(advantages), t(values[:-1]),
                t(values2[:-1])
                )

    def calculate_loss(
            self, states, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        ne_step = s
        """
        if self.training_aup:
            model = self.model_aup
        else:
            model = self.model
        values, policy = model(states)
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
        if self.training_aup:
            entropy_loss *= -self.entropy_aup
        else:
            entropy_loss *= -self.entropy_reg

        return entropy, policy_loss + value_loss * self.vf_coef + entropy_loss

    def train_batch(self, batch):
        # batch = self.gen_training_batch(self.steps_per_env)
        idx = np.arange(len(batch.states))

        for _ in range(self.epochs_per_batch):
            np.random.shuffle(idx)
            for k in idx.reshape(self.num_minibatches, -1):
                entropy, loss = self.calculate_loss(
                    batch.states[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
                if self.training_aup:
                    optimizer = self.optimizer_aup
                else:
                    optimizer = self.optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
            if n > (self.train_aup_steps*self.idx) and not self.switch_to_ppo:
                self.switch_to_ppo = True
                print ('---------------------------------')
                print ('Training Impact Aware Agent')
                self.training_aup = False
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model', 'optimizer', 'model_aup', 'optimizer_aup',
                    ])

            if n >= next_report and self.summary_writer is not None:
                writer = self.summary_writer
                entropy, loss = self.calculate_loss(
                    batch.states, batch.actions, batch.action_prob,
                    batch.values, batch.returns, batch.advantages)
                loss = loss.item()
                entropy = entropy.mean().item()
                values = batch.values.mean().item()
                values2 = batch.values2.mean().item()
                advantages = batch.advantages.mean().item()
                try:
                    rrnd = self.rand_rewards.mean().item()
                except:
                    rrnd = 0
                try:
                    scale = self.scale.mean().item()
                    lamb = self.lamb
                    penalty = self.penalty.item()
                except:
                    scale = 0
                    lamb = 0
                    penalty = 0
                logger.info(
                    "n=%i: loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g, rrwd=%0.3f",
                    n, loss, entropy, values, advantages, rrnd)
                writer.add_scalar("training/{}/loss".format(self.idx), loss, n)
                writer.add_scalar("training/{}/entropy".format(self.idx), entropy, n)
                writer.add_scalar("training/{}/values".format(self.idx), values, n)
                writer.add_scalar("training/{}/values2".format(self.idx), values2, n)
                writer.add_scalar("training/{}/advantages".format(self.idx), advantages, n)
                if self.training_aup:
                    writer.add_scalar("training/{}/rand_reward".format(self.idx), rrnd, n)
                if not self.training_aup:
                    writer.add_scalar("training/{}/scale_value".format(self.idx), scale, n)
                    writer.add_scalar("training/{}/lambda".format(self.idx), lamb, n)
                    writer.add_scalar("training/{}/aup_penalty".format(self.idx), penalty, n)
                writer.flush()

            if n >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model', 'optimizer', 'model_aup', 'optimizer_aup',
                ])

            if n >= next_test:
                self.run_test_envs()
