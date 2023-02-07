import sys
import gym
import time
import numpy as np

from moab_env import MoabEnv, MoabDomainRandEnv
from sb3_contrib import RecurrentPPO

env = MoabDomainRandEnv()
model = RecurrentPPO.load(sys.argv[1])


# phys_params = {
#     "dt": 0.045,  # in s, 45ms
#     "gravity": 3,  # m/s^2, Earth: there's no place like it.
#     "plate_radius": 0.225 / 2,  # m, Moab: 225mm dia
#     "ball_mass": 0.004,  # kg, Ping-Pong ball: 2.7g
#     "ball_radius": 0.025,  # m, Ping-Pong ball: 20mm
#     "ball_shell": 0.0002,  # m, Ping-Pong ball: 0.2mm
#     "max_starting_velocity": 1.0,  # m/s, Ping-Pong ball: 1.0m/s
# }
# env.sim.params = phys_params


obs = env.reset()
r_tot = 0

# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

while True:
    for _ in range(100):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        obs, reward, done, info = env.step(action)
        r_tot += reward

        env.render()

        if done:
            print("Ep reward:", r_tot)
            obs = env.reset()
            print(env.sim.params)
            r_tot = 0
    obs = env.reset()
    print("Ep reward:", r_tot)
    r_tot = 0

env.close()
