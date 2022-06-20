import sys
import gym
from moab_env import MoabEnv
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


save_path = "./logs/" + sys.argv[1]
save_path = save_path[:-1] if save_path[-1] == "/" else save_path

tb_path = "./tb/" + sys.argv[1]
env = MoabEnv()
env = TimeLimit(env, max_episode_steps=2048)

# fmt: off
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=save_path, name_prefix="moab")
# fmt: on
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_path)
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
model.save(save_path + "/trained_moab")


env = MoabEnv()
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
