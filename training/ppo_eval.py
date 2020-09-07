import logging
import numpy as np

import torch
import torch.optim as optim
import torchvision

from safelife.helper_utils import load_kwargs
from safelife.render_graphics import render_board

from .utils import named_output, round_up, LinearSchedule
from . import checkpointing
from .cb_vae import train_encoder, load_state_encoder, encode_state
import matplotlib.pyplot as plt
import os

import matplotlib.lines as lines
import seaborn as sns
import pandas as pd
import matplotlib.patheffects as PathEffects

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


    def __init__(self, ppo_agent, model, model_aup, env_type, z_dim, name, level_idx, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model_aux = model.to(self.compute_device)
        self.model_aup = model_aup.to(self.compute_device)
        self.level = level_idx
        self.name = name
        self.optimizer_aux = optim.Adam(self.model_aux.parameters(), lr=self.learning_rate)
        self.optimizer_aup = optim.Adam(self.model_aup.parameters(), lr=self.learning_rate_aup)
        print (self.logdir)
        checkpointing.load_checkpoint(self.logdir[:-12], self, aux=True)
        checkpointing.load_checkpoint(self.logdir[:-12], self, aup=True)
        self.model_aux.eval()
        self.model_aup.eval()

        self.exp = env_type
        """ AUP-specific parameters """
        self.z_dim = z_dim
        self.state_encoder = None
        self.use_scale = False
        self.steps = 0
        self.user_input = False
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
            if self.steps > 150 and self.user_input==True:
                action = input('action: ')
                if action == 'play':
                    self.user_input = False
                    action = np.random.choice(len(policy), p=policy) # fine 
                else:
                    try:
                        action = int(action)
                    except:
                        print ('Bad action input')
                        action = 0
            else:
                action = np.random.choice(len(policy), p=policy) # fine 
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
            #plt.imshow(self.obsp)
            #os.makedirs('gifs/trajectories/{}/{}/{}'.format(
            #    self.exp, self.name, self.level), exist_ok=True)
            #plt.savefig('gifs/trajectories/{}/{}/{}/{}'.format(
            #    self.exp, self.name, self.level, self.steps))
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
        values_q_aup, policies_aup = self.model_aup(tensor_states)
        values = values_q_aup.mean(1)
        values_q_aux, policies_aux = self.model_aux(tensor_states)
        
        values = values.detach().cpu().numpy()
        policies = policies_aup.detach().cpu().numpy()
        
        actions = []
        rewards = []
        dones = []
        obsp = True
        for i, (policy, env) in enumerate(zip(policies, envs)):

            if self.user_input==True:
                action = input('action: ')
                if action == 'play':
                    self.user_input = False
                    action = np.random.choice(len(policy), p=policy) # fine 
                else:
                    try:
                        action = int(action)
                    except:
                        print ('Bad action input')
                        action = 0
            else:
                action = np.random.choice(len(policy), p=policy) # fine 
            #if self.steps > 50 and self.obsp is not None:
            #    plt.imshow(self.obsp)
            #    plt.show()
            #action = np.random.choice(len(policy), p=policy) # fine 
            print (action, 'taken')
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
            env.last_obs = obs
            actions.append(action)

            # calculate AUP penalty
            noop_value = values_q_aux[i, 0]
            max_value = values_q_aux[i, action]
            penalty = (max_value - noop_value).abs()
            advantages = values_q_aux[i] - noop_value
            self.scale = noop_value
            
            if self.use_scale:
                scale = noop_value
            else:
                scale = 1.
            #lamb = self.lamb_schedule.value(self.num_steps-self.train_aup_steps)
            lamb = 0.1
            self.lamb = lamb
            
            self.penalty = penalty
            reward = reward - lamb * (penalty / scale)
            a_names = ['noop', 'up', 'right', 'down', 'left', 'toggle up', 'toggle right', 'toggle down', 'toggle left']
            print ('STEP {}'.format(self.steps))
            for i, label in enumerate(a_names):
                print (label)
                print ('\t policy: {}'.format(policy[i]))
        

            self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
            plt.close('all')
            plt.imshow(self.obsp)
            plt.show()
            
            reward = reward.cpu().tolist()
            rewards.append(reward)
            dones.append(done)
            self.steps += 1
        
        # print (rewards)
        return states, actions, rewards, dones, policies, values

    def run_test(self, envs):
        obsp = None
        self.steps = 0
        self.done = False
        sns.set()
        mean_actions = np.zeros(9)
        for env in envs:
            while not self.done:
                states = [
                        e.last_obs if hasattr(e, 'last_obs') else e.reset()
                        for e in envs
                        ]
                tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
                values_q, policies = self.model_aup(tensor_states)
                values_aux, policies_aux = self.model_aux(tensor_states)
                policy = policies.detach().cpu().numpy()[0]

                action = np.random.choice(len(policy), p=policy) # fine 
                obs, reward, done, info = env.step(action)

                noop_value = values_aux[0][0]
                max_value = values_aux[0][action]
                penalty = (max_value - noop_value).abs()
                advantages = values_aux[0] - noop_value
                self.scale = noop_value
                a_names = ['noop', 'up', 'right', 'down', 'left',
                        'toggle up', 'toggle right', 'toggle down', 'toggle left']
                print ('STEP {}'.format(self.steps))
                for i, label in enumerate(a_names):
                    print (label)
                    print ('\t policy: {}'.format(policy[i]))
                    print ('\t value_aux: {}'.format(values_aux[0][i]))
                    print ('\t advantage_aux: {}'.format(advantages[i]))
                    print()
                
                chosen = action  # The action chosen by AUP
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
                vmax, vmin = .02, -.02

                actions = [r'$\uparrow$', r'$\rightarrow$', r'$\downarrow$', r'$\leftarrow$',
                        r'$cu$', r'$cr$', r'$cd$', r'$cl$']
                l_q = advantages[1:].cpu().numpy()
                means = np.array([0., 2.21406718, 2.21948796, -0.40617021,  0.66383862,  4.50202886, 0.84547208, 1.26441046,  2.4654113 ])
                l_q -= means[1:]

                mean_actions += advantages.cpu().numpy()
                # l_q -= l_q_mean
                

                vmax, vmin = .02, -.02  # Color max/min values
                data = pd.DataFrame(enumerate(l_q), columns=["x", "y"])  # load list of Q-values
                v = data.y.values
                print (data.y.values, data.x.values)

                cmap = sns.diverging_palette(10, 133, s=50, as_cmap=True)  # get red-white-green colormap
                colors = cmap((v - vmin) / (vmax - vmin))  # normalize bounds

                ax1 = sns.barplot("x", y="y", data=data, palette=colors, ax=ax1)
                ax1.get_xaxis().tick_bottom()
                ax1.axes.get_yaxis().set_visible(False)
                ax1.axes.get_xaxis().set_visible(False)
                ax1.set_facecolor("white")

                xmin, xmax = ax1.get_xaxis().get_view_interval()
                ax1.add_artist(lines.Line2D((xmin, xmax), (0, 0), color='black', linewidth=2))  # set baseline for no-op

                size = 29
                voffset = .05 * vmax
                for i in range(len(actions)):
                    txt = ax1.text(i - .01, voffset, actions[i], horizontalalignment='center', fontsize=size)
                    txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
                """
                l_aup = (values_q[0] - values_q[0][0])[1:].cpu().numpy()
                vmax, vmin = .02, -.02  # Color max/min values
                data = pd.DataFrame(enumerate(l_aup), columns=["x", "y"])  # load list of Q-values
                v = data.y.values
                cmap = sns.diverging_palette(10, 133, s=50, as_cmap=True)  # get red-white-green colormap
                colors = cmap((v - vmin) / (vmax - vmin))  # normalize bounds
                
                ax3 = sns.barplot("x", y="y", data=data, palette=colors, ax=ax3)
                ax3.get_xaxis().tick_bottom()
                ax3.axes.get_yaxis().set_visible(False)
                ax3.axes.get_xaxis().set_visible(False)
                ax3.set_facecolor("white")
                xmin, xmax = ax3.get_xaxis().get_view_interval()
                ax3.add_artist(lines.Line2D((xmin, xmax), (0, 0), color='black', linewidth=2))  # set baseline for no-op
                size = 29
                voffset = .05 * vmax
                for i in range(len(actions)):
                    txt = ax3.text(i - .01, voffset, actions[i], horizontalalignment='center', fontsize=size)
                    txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
                """
                self.obsp = render_board(env.game.board, env.game.goals, env.game.orientation)
                ax2.imshow(self.obsp)
                ax2.grid(False)
                #plt.show()
                os.makedirs('mp4_results/', exist_ok=True)
                plt.savefig('mp4_results/step_{}.png'.format(self.steps))
                plt.close('all')
                
                if done:
                    obs = env.reset()
                env.last_obs = obs
                self.steps += 1
                print ('steps: ', self.steps)
                if done:
                    print (mean_actions / self.steps)
                    return


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
