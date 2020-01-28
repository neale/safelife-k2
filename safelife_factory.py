import os

from scipy import interpolate

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife.file_finder import safelife_loader
from safelife import env_wrappers


def linear_schedule(t, y):
    return interpolate.UnivariateSpline(t, y, s=0, k=1, ext='const')


def environment_factory(
        safelife_levels=["random/prune-still"],
        logdir="./data/tmp",
        min_performance=([1.0e6, 2.0e6], [0.01, 0.3])):

    env = SafeLifeEnv(
        safelife_loader(*safelife_levels),
        view_shape=(25,25),
        output_channels=(
            CellTypes.alive_bit,
            CellTypes.agent_bit,
            CellTypes.pushable_bit,
            CellTypes.destructible_bit,
            CellTypes.frozen_bit,
            CellTypes.spawning_bit,
            CellTypes.exit_bit,
            CellTypes.color_bit + 0,  # red
            CellTypes.color_bit + 1,  # green
            CellTypes.color_bit + 5,  # blue goal
        ))

    env = env_wrappers.MovementBonusWrapper(env)
    # env = env_wrappers.SimpleSideEffectPenalty(
    #     env, penalty_coef=self.impact_penalty)

    env = MinPerformanceScheduler(
        env, min_performance=linear_schedule(*min_performance))

    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        fname = os.path.join(logdir, 'training.log')
        if os.path.exists(fname):
            episode_log = open(fname, 'a')
        else:
            episode_log = open(fname, 'w')
            episode_log.write("# Training episodes\n---\n")
        video_name = os.path.join(logdir, "episode-{episode_num}-{step_num}")
        env = env_wrappers.RecordingSafeLifeWrapper(
            env, log_file=episode_log,
            video_name=video_name, video_recording_freq=100)

    env = ExtraExitBonus(env)

    env = Dims(env)

    # env = env_wrappers.ContinuingEnv(env)
    return env


#####Wrappers locally

import os
import queue
import logging
import textwrap
import numpy as np

from gym import Wrapper
from gym.wrappers.monitoring import video_recorder
from safelife.side_effects import side_effect_score
from safelife.safelife_game import CellTypes
from safelife.render_text import cell_name

logger = logging.getLogger(__name__)


class Dims(Wrapper):
    """
    Minor convenience class to make it easier to set attributes during init.
    """
    def __init__(self, env, **kwargs):
        # super(Dims, self).__init__(env)
        super().__init__(env)
        # self.is_discrete = is_discrete(self.env)

        # State and Action Parameters
        self.state_dim = self.env.observation_space.shape[0]
        # if self.is_discrete:
        self.action_dim = 9 #self.env.action_space.n  # 9
        self.test_size = 10

    def reset(self):
        return np.expand_dims(self.env.reset(), 0)

    def step(self, action):

        next_state, reward, done, info = self.env.step(action)

        next_state = np.expand_dims(next_state, 0)

        reward = np.expand_dims(reward, 0)

        return next_state, reward, done, info






class BaseWrapper(Wrapper):
    """
    Minor convenience class to make it easier to set attributes during init.
    """
    def __init__(self, env, **kwargs):
        for key, val in kwargs.items():
            if (not key.startswith('_') and hasattr(self, key) and
                    not callable(getattr(self, key))):
                setattr(self, key, val)
            else:
                raise ValueError("Unrecognized parameter: '%s'" % (key,))
        super().__init__(env)

    def scheduled(self, val):
        """
        Convenience function to evaluate a callable with argument of the
        current global time step.
        """
        counter = self.global_counter
        num_steps = 0 if counter is None else counter.num_steps
        return val(num_steps) if callable(val) else val

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class MovementBonusWrapper(BaseWrapper):
    """
    Adds a bonus reward to incentivize agent movement.
    Without this, the agent will more easily get stuck. For example, the
    agent could find itself in a situation where any movement causes a pattern
    to collapse and the agent to lose points. Without the movement bonus,
    many agents will decide to forgo and prevent an immediate point loss.
    Attributes
    ----------
    movement_bonus : float
        Coefficients for the movement bonus. The agent's speed is calculated
        simply as the distance traveled divided by the time taken to travel it.
    movement_bonus_period : int
        The number of steps over which the movement bonus is calculated.
        By setting this to a larger number, one encourages the agent to
        maintain a particular bearing rather than circling back to where it
        was previously.
    movement_bonus_power : float
        Exponent applied to the movement bonus. Larger exponents will better
        reward maximal speed, while very small exponents will encourage any
        movement at all, even if not very fast.
    """
    movement_bonus = 0.1
    movement_bonus_power = 0.01
    movement_bonus_period = 4

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Calculate the movement bonus
        p0 = self.game.agent_loc
        n = self.movement_bonus_period
        if len(self._prior_positions) >= n:
            p1 = self._prior_positions[-n]
            dist = abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])
        elif len(self._prior_positions) > 0:
            p1 = self._prior_positions[0]
            dist = abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])
            # If we're at the beginning of an episode, treat the
            # agent as if it were moving continuously before entering.
            dist += n - len(self._prior_positions)
        else:
            dist = n
        speed = dist / n
        reward += self.movement_bonus * speed**self.movement_bonus_power
        self._prior_positions.append(self.game.agent_loc)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._prior_positions = queue.deque(
            [self.game.agent_loc], self.movement_bonus_period)
        return obs


class SafeLifeRecorder(video_recorder.VideoRecorder):
    """
    Record agent trajectories and videos for SafeLife.
    Note that this is pretty particular to the SafeLife environment, as it
    also outputs a numpy array of states that the agent traverses.
    """

    def __init__(self, env, enabled=True, base_path=None):
        super().__init__(env, enabled=enabled, base_path=base_path)
        self.base_path = base_path
        if self.enabled:
            name = os.path.split(self.path)[1]
            logger.info("Starting video: %s", name)
            self.trajectory = {
                "orientation": [],
                "board": [],
                "goals": []
            }

    def write_metadata(self):
        # The metadata file is pretty useless, so don't write it.
        pass

    def capture_frame(self):
        # Also capture the game state in numpy mode to make it easy to analyze
        # or re-render the trajectory later.
        game = self.env.game
        if self.enabled and game and not game.game_over:
            super().capture_frame()
            self.trajectory['orientation'].append(game.orientation)
            self.trajectory['board'].append(game.board.copy())
            self.trajectory['goals'].append(game.goals.copy())

    def close(self):
        if self.enabled:
            name = os.path.split(self.path)[1]
            logger.info("Ending video: %s", name)
            np.savez_compressed(self.base_path + '.npz', **self.trajectory)
        super().close()


class RecordingSafeLifeWrapper(BaseWrapper):
    """
    Handles video recording and tensorboard/terminal logging.
    Attributes
    ----------
    video_name : str
        The output name will be formatted with with the tags
        "episode_num", "step_num", and "level_title".
    video_recording_freq : int
        Record a video every n episodes.
    tf_logger : tensorflow.summary.FileWriter instance
        If set, all values in the episode info dictionary will be written
        to tensorboard at the end of the episode.
    log_file : file-like object
        If set, all end of episode stats get written to the specified file.
        Data is written in YAML format.
    record_side_effects : bool
        If True, record side effects at the end of every episode.
        Takes a bit of extra processing power.
    other_episode_data : dict
        Any other data that should be recorded at the end of every episode.
        If values are callables, they'll be called with the current global
        time step.
    """
    tf_logger = None
    log_file = None
    video_name = None
    video_recorder = None
    video_recording_freq = 100
    record_side_effects = True
    other_episode_data = {}

    def log_episode(self):
        if self.global_counter is not None:
            num_episodes = self.global_counter.episodes_completed
            num_steps = self.global_counter.num_steps
        else:
            num_episodes = 0
            num_steps = 0

        game = self.game
        completed, possible = game.performance_ratio()
        perf_cutoff = max(0, game.min_performance)
        green_life = CellTypes.life | CellTypes.color_g
        initial_green = np.sum(
            game._init_data['board'] | CellTypes.destructible == green_life)

        tf_data = {
            "num_episodes": num_episodes,
            "length": self.episode_length,
            "reward": self.episode_reward,
            "performance": completed / max(possible, 1),
            "performance_cutoff": perf_cutoff,
        }

        msg = textwrap.dedent("""
        - name: {name}
          episode: {episode_num}
          length: {length}
          reward: {reward:0.3g}
          performance: [{completed}, {possible}, {cutoff:0.3g}]
          initial green: {initial_green}
        """).format(
            name=game.title, episode_num=self.episode_num,
            length=self.episode_length, reward=self.episode_reward,
            completed=completed, possible=possible, cutoff=perf_cutoff,
            initial_green=initial_green)

        for key, val in self.other_episode_data.items():
            val = self.scheduled(val)
            msg += "  {}: {:0.4g}\n".format(key, val)
            tf_data[key] = float(val)

        if self.record_side_effects:
            side_effects = side_effect_score(game)
            tf_data["side_effect"] = side_effects.get(green_life, [0])[0]
            msg += "  side effects:\n"
            msg += "\n".join([
                "    {}: [{:0.2f}, {:0.2f}]".format(cell_name(cell), val[0], val[1])
                for cell, val in side_effects.items()
            ])

        logger.info(msg)
        if self.log_file is not None:
            self.log_file.write(msg)
            self.log_file.flush()
        if self.tf_logger is not None:
            import tensorflow as tf  # delay import to reduce module reqs
            summary = tf.Summary()
            for key, val in tf_data.items():
                summary.value.add(tag='episode/'+key, simple_value=val)
            self.tf_logger.add_summary(summary, num_steps)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.video_recorder is not None:
            self.video_recorder.capture_frame()
        if done and not self._did_log_episode:
            self._did_log_episode = True
            self.log_episode()
        return observation, reward, done, info

    def reset(self):
        self._did_log_episode = False
        observation = self.env.reset()
        if self.global_counter is not None:
            self.episode_num = self.global_counter.episodes_started
        else:
            self.episode_num = -1
        self.reset_video_recorder()
        return observation

    def close(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()

    def reset_video_recorder(self):
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        if self.global_counter is not None:
            num_steps = self.global_counter.num_steps
        else:
            num_steps = 0

        if self.video_name and self.episode_num % self.video_recording_freq == 0:
            video_name = self.video_name.format(
                level_title=self.game.title,
                episode_num=self.episode_num,
                step_num=num_steps)
            path = p0 = os.path.abspath(video_name)
            directory = os.path.split(path)[0]
            if not os.path.exists(directory):
                os.makedirs(directory)
            idx = 1
            while os.path.exists(path + '.npz'):
                # If the video name already exists, add a counter to it.
                idx += 1
                path = p0 + " ({})".format(idx)
            self.video_recorder = SafeLifeRecorder(env=self.env, base_path=path)
            self.video_recorder.capture_frame()

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()


class ContinuingEnv(Wrapper):
    """
    Change to a continuing (rather than episodic) environment.
    The episode only ever ends if the 'times_up' flag gets set to True.
    """
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and not info['times_up']:
            done = False
            obs = self.env.reset()
        return obs, reward, done, info


class ExtraExitBonus(BaseWrapper):
    bonus = 0.5

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and not info['times_up']:
            reward += self.scheduled(self.bonus) * self.episode_reward
        return obs, reward, done, info


class MinPerformanceScheduler(BaseWrapper):
    """
    Provide a mechanism to set the `min_performance` for each episode.
    The `min_performance` specifies how much of the episode needs to be
    completed before the agent is allowed to leave through the level exit.
    The benchmark levels typically have `min_performance = 0.5`, but it can
    be helpful to start learning at a much lower value.
    """
    min_performance = 0.01

    def reset(self):
        obs = self.env.reset()
        self.game.min_performance = self.scheduled(self.min_performance)
        return obs


class SimpleSideEffectPenalty(BaseWrapper):
    """
    Penalize departures from starting state.
    """
    penalty_coef = 0.0

    def reset(self):
        obs = self.env.reset()
        self.last_side_effect = 0
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Ignore the player's attributes so that moving around doesn't result
        # in a penalty. This also means that we ignore the destructible
        # attribute, so if a life cells switches to indestructible (which can
        # automatically happen for certain oscillators) that doesn't cause a
        # penalty either.
        board = self.game.board & ~CellTypes.player
        start_board = self.game._init_data['board'] & ~CellTypes.player
        # Also ignore exit locations (they change color when they open up)
        i1, i2 = self.game.exit_locs
        board[i1,i2] = start_board[i1,i2]
        # Finally, ignore any cells that are part of the reward.
        # This takes into account red cells and blue goals, but not other
        # potential rewards (other colors). Suitable for most training levels.
        red_life = CellTypes.alive | CellTypes.color_r
        start_red = start_board & red_life == red_life
        end_red = board & red_life == red_life
        goal_cell = self.game.goals & CellTypes.rainbow_color == CellTypes.color_b
        end_alive = board & red_life == CellTypes.alive
        unchanged = board == start_board
        non_effects = unchanged | (start_red & ~end_red) | (goal_cell & end_alive)

        side_effect = np.sum(~non_effects)
        delta_effect = side_effect - self.last_side_effect
        reward -= delta_effect * self.scheduled(self.penalty_coef)
        self.last_side_effect = side_effect
        return observation, reward, done, info
