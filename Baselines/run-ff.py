import sys
import gym
import time
from moab_env import MoabEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

env = MoabEnv()
model = PPO.load(sys.argv[1])


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

n_repeat = 3

obs = env.reset(config={"dt": 0.0333 / n_repeat})
r_tot = 0

while True:
    for _ in range(300):
        action, _states = model.predict(obs, deterministic=True)

        ep_len = 0
        for _ in range(n_repeat):
            ep_len += 1
            obs, reward, done, info = env.step(action)
            r_tot += reward
            env.render()

            if done:
                break
        # time.sleep(0.5)

        if done:
            obs = env.reset()
            print("Ep len:", ep_len, "Ep reward:", r_tot / n_repeat)
            r_tot = 0
    obs = env.reset()
    print("Ep reward:", r_tot / n_repeat)
    r_tot = 0

env.close()
