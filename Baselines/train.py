import sys
import gym

from moab_env import MoabEnv, MoabDomainRandEnv, MoabPartialDomainRandEnv
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.utils import get_schedule_fn


ENVS = {
    "MoabEnv": MoabEnv,
    "MoabDomainRandEnv": MoabDomainRandEnv,
    "MoabPartialDomainRandEnv": MoabPartialDomainRandEnv,
}


def train(
    run_name,
    num_timesteps,
    checkpoint_frequency=10_000,
    render_checkpoints=False,
    play_when_done=False,
    env_name="MoabDomainRandEnv",
    lstm=True,
    enable_critic_lstm=True,
    shared_lstm=False,
    reset_hidden=True,
    env_params={},
):
    run_name = run_name[:-1] if run_name[-1] == "/" else run_name
    save_path = "./logs/" + run_name
    tb_path = "./tb/" + run_name

    env = ENVS[env_name](**env_params)
    eval_env = ENVS[env_name](**env_params)
    env = TimeLimit(env, max_episode_steps=2048)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency, save_path=save_path, name_prefix="moab"
    )

    if render_checkpoints:
        eval_callback = EvalCallback(
            eval_env, eval_freq=checkpoint_frequency, render=True
        )
        callback = CallbackList([checkpoint_callback, eval_callback])
    else:
        callback = checkpoint_callback

    if lstm:
        policy_params = {
            "lstm_hidden_size": 64,  # Reduced from 256
            "n_lstm_layers": 1,  # Default
            # (below) False by default (on would be only use gradient from actor)
            "shared_lstm": shared_lstm,
            "enable_critic_lstm": enable_critic_lstm,
            "reset_hidden": reset_hidden,
            "lstm_kwargs": None,
        }

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_path,
            policy_kwargs=policy_params,
        )
        model.learn(
            total_timesteps=num_timesteps,
            callback=checkpoint_callback,
            tb_log_name="first_run",
        )
        model.save(save_path + "/trained_moab")

    else:
        policy_params = {
            # lr_schedule: Schedule
            # net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None
            # activation_fn: Type[nn.Module] = nn.Tanh
            # ortho_init: bool = True
            # use_sde: bool = False
            # log_std_init: float = 0.0
            # full_std: bool = True
            # sde_net_arch: Optional[List[int]] = None
            # use_expln: bool = False
            # squash_output: bool = False
            # features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor
            # features_extractor_kwargs: Optional[Dict[str, Any]] = None
            # normalize_images: bool = True
            # optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam
            # optimizer_kwargs: Optional[Dict[str, Any]] = None
        }

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_path,
            policy_kwargs=policy_params,
        )
        model.learn(
            total_timesteps=num_timesteps,
            callback=callback,
            tb_log_name="first_run",
        )
        model.save(save_path + "/trained_moab")

    if play_when_done:
        try:
            env = MoabEnv()
            obs = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
        finally:  # Close on ctrl-c
            env.close()


def append_timesteps_str(run_name, num_timesteps):
    # Append the timesteps amount to the run name ie `-1.5m` for 1.5 million timesteps
    if num_timesteps < 1_000_000:
        run_name = run_name + f"-{int(num_timesteps / 1000)}k"
    else:
        run_name = run_name + f"-{int(num_timesteps / 1_000_000)}"
        # Add the .5 if it's something like 1.5 million
        remainder = num_timesteps / 1_000_000 - int(num_timesteps / 1_000_000)

        if remainder != 0:
            run_name += f".{int(remainder * 10)}m"
        else:
            run_name += "m"

    return run_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-frequency", default=10_000, type=int)
    parser.add_argument("-p", "--play-when-done", default=False, type=bool)
    parser.add_argument("-r", "--render-checkpoints", default=False, type=bool)
    parser.add_argument("-n", "--num-timesteps", default=1_000_000, type=float)
    args, _ = parser.parse_known_args()

    num_timesteps = int(args.num_timesteps)

    runs = {
        "reference-no-lstm": {"lstm": False},
        "no-lstm-newenv": {"env_name": "MoabEnv", "lstm": False},
        "dr-ff": {"env_name": "MoabDomainRandEnv", "lstm": False},
        "reference-no-dr": {"env_name": "MoabEnv"},
        "dr": {},
        "dr-no-reset": {"reset_hidden": False},
        "dr-critic-no-lstm": {"enable_critic_lstm": False},
        "dr-critic-shared-lstm": {"shared_lstm": True, "enable_critic_lstm": False},
    }

    for run_name, params in runs.items():
        run_name = append_timesteps_str(run_name, num_timesteps)
        train(
            run_name,
            int(args.num_timesteps),
            args.checkpoint_frequency,
            args.render_checkpoints,
            args.play_when_done,
            **params,
        )
