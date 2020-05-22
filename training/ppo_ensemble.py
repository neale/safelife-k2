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


class PPO_AUP(object):
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
    learning_rate_aup = 1e-4
    entropy_reg = 0.001 #0.01

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


    def __init__(self, model, models_aup, env_type, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model = model.to(self.compute_device)
        self.models_aup = [m.to(self.compute_device) for m in models_aup]
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.optimizers_aup = [optim.Adam(m.parameters(),
                               lr=self.learning_rate_aup) for m in self.models_aup]
        checkpointing.load_checkpoint(self.logdir, self)
        self.exp = env_type

        """ AUP-specific parameters """
        self.state_encoder = None
        self.n_random_reward_fns = 1
        self.training_aup = True    # indicator: currently training random reward agent
        self.impact_training = True # indicator: use penalty (True) or not (False) in take_one_step
        self.switch_to_ppo = False  # indicator: turn on PPO (True) or off (False)
        self.train_aup_steps=2.5e5
        if self.impact_training:
            self.lamb_schedule = LinearSchedule(2e6, initial_p=1e-8, final_p=1e-3)
            self.load_rendered_state_buffer = True
            self.state_encoder_path = 'models/{}/100/model_save_epoch_100.pt'.format(self.exp)
            self.train_state_encoder(envs=self.training_envs)
            for model in self.models_aup:
                model.register_rfn(self.state_encoder.z_dim, self.compute_device)
    
    def get_rand_rewards(self, states, envs, model_aup):
        if not torch.is_tensor(states):
            states = torch.tensor(states)
        if states.dim() != 4:
            states = states.unsqueeze(0)
        states = self.batch_preprocess_states(states, envs)
        self.state_encoder.eval()
        states_z = encode_state(self.state_encoder, states, self.compute_device)
        rewards = torch.mm(states_z, model_aup.random_fn.T)
        return rewards
    
    def batch_preprocess_states(self, states, envs):
        obs = [render_board(env.game.board, env.game.goals, env.game.orientation) for env in envs]
        obs = np.asarray(obs)
        obs = torch.from_numpy(np.matmul(obs[:, :, :, :3], [0.299, 0.587, 0.114]))
        obs = obs.unsqueeze(0) # [1, batch, H, W]
        obs = F.avg_pool2d(obs, 5, 4)/255.
        return obs.float()

    def train_state_encoder(self, envs, buffer_size=100e3):
        if self.state_encoder_path is not None:
            print ('loading state encoder')
            self.state_encoder = load_state_encoder(z_dim=100, path=self.state_encoder_path, device=self.compute_device)
            return
        if self.load_rendered_state_buffer:
            #buffer_np = np.load('buffers/prune-still/state_buffer.npy')
            buffer_np = np.load('buffers/{}/append-still/state_buffer.npy'.format(self.exp))
            buffer_th = torch.from_numpy(buffer_np).float()
        else:
            envs = [e.unwrapped for e in envs]
            states = [
                e.last_obs if hasattr(e, 'last_obs') else e.reset() for e in envs
            ]
            print ('gathering data from every env N={}'.format(len(envs)))
            buffer = []
            buffer_size = int(buffer_size // len(envs))
            for env in envs:
                for _ in range(buffer_size):
                    action = np.random.choice(env.action_space.n, 1)[0]
                    obs, _, done, _ = env.step(action)
                    if done:
                        obs = env.reset()
                    obs = render_board(env.game.board, env.game.goals, env.game.orientation)
                    obs = torch.tensor(np.dot(obs[...,:3], [0.299, 0.587, 0.114]))
                    obs = obs.unsqueeze(0).unsqueeze(0)
                    obs = F.avg_pool2d(obs, 5, 4).squeeze(0).squeeze(0)/255.
                    buffer.append(obs)
            print ('collected data')
            buffer_th = torch.stack(buffer)
            buffer_np = buffer_th.cpu().numpy()
            np.save('buffers/{}/state_buffer'.format(self.exp), buffer_np)
            # np.save('buffers/prune-still/state_buffer', buffer_np)
        self.state_encoder = train_encoder(device=self.compute_device, data=buffer_th, z_dim=100, training_epochs=100, exp=self.exp)
        return

    @named_output('states actions rewards done policies values')
    def take_one_step(self, envs, model):
        states = [
            e.last_obs if hasattr(e, 'last_obs') else e.reset()
            for e in envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values_q, policies = self.model(tensor_states)
        values = values_q.mean(1)
        
        val_aup = []
        for model_aup in self.models_aup:
            values_q_aup, policies_aup = model_aup(tensor_states)
            val_aup.append(values_q_aup)

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
                penalty = 0.
                scale = 0.
                for values_q_aup in val_aup:
                    noop_value = values_q_aup[i, 0]
                    max_value = values_q_aup[i].max()
                    penalty += (max_value - noop_value).abs()
                    scale += noop_value
                lamb = self.lamb_schedule.value(self.num_steps-self.train_aup_steps)
                reward = reward - lamb * (penalty / scale)
                reward = reward.cpu().tolist()
            rewards.append(reward)
            dones.append(done)
       
        # print (rewards)
        return states, actions, rewards, dones, policies, values

    @named_output('states actions rewards done policies values')
    def take_one_step_rand(self, envs, model_aup):
        states = [
                e.last_obs if hasattr(e, 'last_obs') else e.reset()
                for e in envs
                ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values_q, policies = model_aup(tensor_states)
        values = values_q.mean(1)
        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()

        random_rewards = self.get_rand_rewards(tensor_states, envs, model_aup)
        model_aup.last_rr = random_rewards  # *shrug* for logging
        random_rewards = random_rewards.squeeze(-1).tolist() # [batch, actions]

        actions = []
        rewards = []
        dones = []
        for i, (policy, env) in enumerate(zip(policies, envs)):
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, _, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)
            dones.append(done)

        rewards = random_rewards
        # print ('Batch Rewards: ', np.asarray(rewards).mean())
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

    @named_output('states actions action_prob returns advantages values')
    def gen_training_batch(self, steps_per_env, model, update_n=False, flat=True):
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
            take_one_step = self.take_one_step_rand 
        else:
            take_one_step = self.take_one_step

        steps = [take_one_step(self.training_envs, model) for _ in range(steps_per_env)] # [steps]
        final_states = [e.last_obs for e in self.training_envs]
        tensor_states = torch.tensor(
            final_states, device=self.compute_device, dtype=torch.float32)
        final_vals = model(tensor_states)[0]
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
            return torch.tensor(x, device=self.compute_device, dtype=dtype)

        if update_n:
            self.num_steps += actions.size
        self.num_episodes += np.sum(done)
        return (
            t([s.states for s in steps]), t(actions, torch.int64),
            t(probs), t(returns), t(advantages), t(values[:-1])
        )

    def calculate_loss(
            self, model, states, actions, old_policy, old_values, returns, advantages):
        """
        All parameters ought to be tensors on the appropriate compute device.
        ne_step = s
        """
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
        entropy_loss *= -self.entropy_reg

        return entropy, policy_loss + value_loss * self.vf_coef + entropy_loss

    def train_batch(self, batch, model, optimizer):
        idx = np.arange(len(batch.states))

        for _ in range(self.epochs_per_batch):
            np.random.shuffle(idx)
            for k in idx.reshape(self.num_minibatches, -1):
                entropy, loss = self.calculate_loss(model,
                    batch.states[k], batch.actions[k], batch.action_prob[k],
                    batch.values[k], batch.returns[k], batch.advantages[k])
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

            n = self.num_steps
            if n > self.train_aup_steps and not self.switch_to_ppo:
                self.switch_to_ppo = True
                print ('---------------------------------')
                print ('Training Impact Aware Agent')
                self.training_aup = False
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model', 'optimizer'])#, 'model_aup', 'optimizer_aup',
                #])
            
            if self.training_aup == True:
                for i, (model, optim) in enumerate(zip(self.models_aup, self.optimizers_aup)):
                    batch = self.gen_training_batch(self.steps_per_env,
                                                    model=model, update_n=(i==(len(self.models_aup)-1)))
                    self.train_batch(batch, model, optimizer=optim)
                    #if not n >= next_report and self.summary_writer is not None:
                    # print ('writing report')
                    writer = self.summary_writer
                    entropy, loss = self.calculate_loss(model, 
                            batch.states, batch.actions, batch.action_prob,
                            batch.values, batch.returns, batch.advantages)
                    loss = loss.item()
                    entropy = entropy.mean().item()
                    values = batch.values.mean().item()
                    advantages = batch.advantages.mean().item()
                    try:
                        rrnd = model.last_rr.mean().item()
                    except: rrnd = 0
                    logger.info(
                            "n=%i: agent=%i, loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g, rrwd=%0.3f",
                            n, i, loss, entropy, values, advantages, rrnd)
                    writer.add_scalar("training/agent{}/loss".format(i), loss, n)
                    writer.add_scalar("training/agent{}/entropy".format(i), entropy, n)
                    writer.add_scalar("training/agent{}/values".format(i), values, n)
                    writer.add_scalar("training/agent{}/advantages".format(i), advantages, n)
                    writer.add_scalar("training/agent{}/rand_reward".format(i), rrnd, n)
                    writer.flush()
            else:
                batch = self.gen_training_batch(self.steps_per_env, model=self.model, update_n=True)
                self.train_batch(batch, model=self.model, optimizer=self.optimizer)
                # if n >= next_report and self.summary_writer is not None:
                writer = self.summary_writer
                entropy, loss = self.calculate_loss(model, 
                        batch.states, batch.actions, batch.action_prob,
                        batch.values, batch.returns, batch.advantages)
                loss = loss.item()
                entropy = entropy.mean().item()
                values = batch.values.mean().item()
                advantages = batch.advantages.mean().item()
                logger.info(
                        "n=%i: loss=%0.3g, entropy=%0.3f, val=%0.3g, adv=%0.3g, rrwd=%0.3f",
                        n, loss, entropy, values, advantages, rrnd)
                writer.add_scalar("training/loss", loss, n)
                writer.add_scalar("training/entropy".format(i), entropy, n)
                writer.add_scalar("training/values".format(i), values, n)
                writer.add_scalar("training/advantages".format(i), advantages, n)
                writer.flush()


            if n >= next_checkpoint:
                checkpointing.save_checkpoint(self.logdir, self, [
                    'model', 'optimizer'])#, 'models_aup', 'optimizers_aup',
                #])

            if n >= next_test:
                self.run_test_envs()
