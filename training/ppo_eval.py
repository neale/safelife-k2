import logging
import numpy as np

import torch
import torch.optim as optim
import torchvision

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from .utils import named_output, round_up, LinearSchedule
from . import checkpointing
from .aae import *
from .cb_vae import train_encoder, load_state_encoder, encode_state
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class PPO_eval(object):
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


    def __init__(self, ppo_agent, model, model_aup, env_type, z_dim, agent_index, name, level_idx, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model = model.to(self.compute_device)
        self.model_aup = model_aup.to(self.compute_device)
        self.idx = agent_index
        self.level = level_idx
        self.name = name
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.optimizer_aup = optim.Adam(
            self.model_aup.parameters(), lr=self.learning_rate_aup)
        checkpointing.load_checkpoint(self.logdir, self)
        self.model.eval()
        self.model_aup.eval()

        self.exp = env_type
        """ AUP-specific parameters """
        self.z_dim = z_dim
        self.state_encoder = None
        self.use_scale = False
        self.steps = 0
        self.user_input = True
        self.obsp = None

    @named_output('states actions rewards done policies values')
    def take_one_step_ppo(self, envs):
        states = [
                e.last_obs if hasattr(e, 'last_obs') else e.reset()
                for e in envs
                ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        print (tensor_states.shape)
        values_q, policies = self.model(tensor_states)
        print (policies.shape)
        values = values_q.mean(1)

        values = values.detach().cpu().numpy()
        policies = policies.detach().cpu().numpy()

        actions = []
        rewards = []
        dones = []
        for i, (policy, env) in enumerate(zip(policies, envs)):
            self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
            #if self.steps > 150 and self.user_input==True:
            #    action = input('action: ')
            #    if action == 'play':
            #        self.user_input = False
            #        action = np.random.choice(len(policy), p=policy) # fine 
            #    else:
            #        try:
            #            action = int(action)
            #        except:
            #            print ('Bad action input')
            #            action = 0
            #else:
            #    action = np.random.choice(len(policy), p=policy) # fine 
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            #actions.append(action)

            # calculate AUP penalty
            #noop_value = values_q[i, 0]
            #max_value = values_q[i, action]
            #penalty = (max_value - noop_value).abs()
            #advantages = values_q[i] - noop_value
            #self.penalty = penalty
            #print ('action: {}, penalty: {}, return: {}, advantages: {}'.format(
            #    action,
            #    penalty,
            #    reward,
            #    advantages))
            #plt.imshow(obsp)
            
            #self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
            plt.imshow(self.obsp)
            os.makedirs('gifs/trajectories/{}/{}/{}'.format(
                self.exp, self.name, self.level), exist_ok=True)
            plt.savefig('gifs/trajectories/{}/{}/{}/{}'.format(
                self.exp, self.name, self.level, self.steps))
            #if self.user_input and self.steps > 150:
            #    plt.imshow(self.obsp)
            #    plt.show()
            rewards.append(reward)
            dones.append(done)
            self.steps += 1
            print (self.steps)
        print (dones)
        return states, actions, rewards, dones, policies, values


    @named_output('states actions rewards done policies values')
    def take_one_step_aup(self, envs):
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
        obsp = None
        for i, (policy, env) in enumerate(zip(policies, envs)):

            #if self.steps > 50 and self.user_input==True:
            #    action = input('action: ')
            #    if action == 'play':
            #        self.user_input = False
            #        action = np.random.choice(len(policy), p=policy) # fine 
            #    else:
            #        try:
            #            action = int(action)
            #        except:
            #            print ('Bad action input')
            #            action = 0
            #else:
            #    action = np.random.choice(len(policy), p=policy) # fine 
            #if self.steps > 50 and self.obsp is not None:
                #plt.imshow(self.obsp)

                #plt.show()
            #    pass
            action = np.random.choice(len(policy), p=policy) # fine 
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            #actions.append(action)

            # calculate AUP penalty
            #noop_value = values_q_aup[i, 0]
            #max_value = values_q_aup[i, action]
            #penalty = (max_value - noop_value).abs()
            #advantages = values_q_aup[i] - noop_value
            #self.scale = noop_value
            #if self.use_scale:
            #    scale = noop_value
            #else:
            #    scale = 1.
            #lamb = self.lamb_schedule.value(self.num_steps-self.train_aup_steps)
            #self.lamb = lamb
            
            #lamb = 0.1
            #self.penalty = penalty
            #reward = reward - lamb * (penalty / scale)
            #a_names = ['noop', 'up', 'right', 'down', 'left', 'toggle up', 'toggle right', 'toggle down', 'toggle left']
            #print ('STEP {}'.format(self.steps))
            #for i, label in enumerate(a_names):
            #    print (label)
            #    print ('\t policy: {}'.format(policy[i]))
            #    print ('\t value_aup: {}'.format(values_q[0][i]))
            #    print ('\t value_aux: {}'.format(advantages[i]))
            #    print()

            self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
            plt.imshow(self.obsp)
            os.makedirs('gifs/trajectories/{}/{}/{}'.format(
                self.exp, self.name, self.level), exist_ok=True)
            plt.savefig('gifs/trajectories/{}/{}/{}/{}'.format(
                self.exp, self.name, self.level, self.steps))
            #os.makedirs('biglevel_trial/trajectories_8/manual_level_{}/'.format(self.level), exist_ok=True)
            #plt.savefig('biglevel_trial/trajectories_8/manual_level_{}/step_{}'.format(self.level, self.steps))
            #if self.user_input and self.steps > 50:
            #    plt.imshow(self.obsp)
            #    plt.show()
            #reward = reward.cpu().tolist()
            rewards.append(reward)
            dones.append(done)
            self.steps += 1
       
        # print (rewards)
        return states, actions, rewards, dones, policies, values

    def run_test(self, envs):
        obsp = None
        path = 'gifs/trajectories/{}/{}/{}'.format(self.exp, self.name, self.level)
        os.makedirs(path, exist_ok=True)
        self.steps = 0
        self.done = False
        for env in envs:
            while not self.done:
                states = [
                        e.last_obs if hasattr(e, 'last_obs') else e.reset()
                        for e in envs
                        ]
                tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
                values_q, policies = self.model(tensor_states)
                policy = policies.detach().cpu().numpy()[0]

                action = np.random.choice(len(policy), p=policy) # fine 
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
                env.last_obs = obs
                self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
                plt.imshow(self.obsp)
                plt.savefig('{}/{}'.format(path, self.steps))
                self.steps += 1
                print ('steps: ', self.steps)
                if done:
                    return
                plt.close('all')


    def test(self):
        """
        Run each testing environment until completion.

        It's assumed that the testing environments take care of their own
        logging via wrappers.
        """
        training_envs = self.training_envs or []
        if self.name in ['ppo', 'naive']:
            test_fn = self.take_one_step_ppo
        else:
            test_fn = self.take_one_step_aup
        with torch.no_grad():
            self.run_test(training_envs)
        return
        while training_envs:
            with torch.no_grad():
                data = test_fn(training_envs)
            training_envs = [
                env for env, done in zip(training_envs, data.done) if not done
            ]
