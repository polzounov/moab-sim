import sys
import time
import gym
import numpy as np
from moab_env import MoabEnv


def create_nn(filepath):
    npz = np.load(filepath)
    w0 = npz["w0"]
    b0 = npz["b0"]
    w1 = npz["w1"]
    b1 = npz["b1"]
    w_out = npz["w_out"]
    return lambda x: w_out @ np.tanh(w1 @ np.tanh(w0 @ x + b0) + b1)


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


env = MoabEnv()
nn = create_nn(sys.argv[1])
obs = env.reset()
r_tot = 0

while True:
    for _ in range(100):
        action = nn(obs)
        obs, reward, done, info = env.step(action)
        r_tot += reward
        time.sleep(0.3)

        env.render()

        if done:
            obs = env.reset()
            print("Ep reward:", r_tot)
            r_tot = 0
    obs = env.reset()
    print("Ep reward:", r_tot)
    r_tot = 0

env.close()
