import sys
import gym
from moab_env import MoabEnv, MoabDomainRandEnv
from gym.wrappers import TimeLimit

# from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn


ENVS = {"MoabEnv": MoabEnv, "MoabDomainRandEnv": MoabDomainRandEnv}


def train(
    run_name,
    num_timesteps=5_000_000,
    env_name="MoabDomainRandEnv",
    enable_critic_lstm=True,
    shared_lstm=False,
    reset_hidden=True,
    env_params={},
):
    run_name = run_name[:-1] if run_name[-1] == "/" else run_name
    save_path = "./logs/" + run_name
    tb_path = "./tb/" + run_name

    env = ENVS[env_name](**env_params)
    env = TimeLimit(env, max_episode_steps=2048)

    # fmt: off
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=save_path, name_prefix="moab")
    # fmt: on

    policy_params = {
        "lstm_hidden_size": 64,  # Reduced from 256
        "n_lstm_layers": 1,  # Default
        "shared_lstm": shared_lstm,  # This is false by default (on would only use gradient from actor)
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

    model.learn(total_timesteps=num_timesteps, callback=checkpoint_callback, tb_log_name="first_run")
    model.save(save_path + "/trained_moab")

    # env = MoabEnv()
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()

    # env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--logs", default="./logs", type=str)
    parser.add_argument("-n", "--num-timesteps", default=1_500_000, type=float)
    args, _ = parser.parse_known_args()

    num_timesteps = int(args.num_timesteps)

    runs = {
        "reference-no-dr": {"env_name": "MoabEnv"},
        "dr": {},
        "dr-no-reset": {"reset_hidden": False},
        "dr-critic-no-lstm": {"enable_critic_lstm": False},
        "dr-critic-shared-lstm": {"shared_lstm": True, "enable_critic_lstm": False},
    }

    for run_name, params in runs.items():
        if num_timesteps < 1_000_000:
            run_name = run_name + f"-{int(num_timesteps / 1000)}k"
        else:
            run_name = run_name + f"-{int(num_timesteps / 1_000_000)}m"

        train(run_name, num_timesteps=int(args.num_timesteps), **params)
