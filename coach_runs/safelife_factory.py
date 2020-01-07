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
    env = env_wrappers.MinPerformanceScheduler(
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
    env = env_wrappers.ExtraExitBonus(env)
    # env = env_wrappers.ContinuingEnv(env)
    return env
