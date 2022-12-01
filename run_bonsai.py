import sys
import gym
import time
import requests
import numpy as np
from moab_env import MoabEnv


# --------------------------------------------------------------------------------------
def brain_controller(port=5555, client_id=123):
    version = 1
    prediction_url = f"http://localhost:{port}/v1/prediction"

    def next_action_v1(state):
        x, y, vel_x, vel_y = state

        observables = {
            "ball_x": float(x),
            "ball_y": float(y),
            "ball_vel_x": float(vel_x),
            "ball_vel_y": float(vel_y),
        }

        action = (0, 0)  # Action is 0,0 if not detected or brain didn't work
        info = {"status": 400, "resp": ""}
        try:
            response = requests.get(prediction_url, json=observables)
            info = {"status": response.status_code, "resp": response.json()}
            if response.ok:
                pitch = info["resp"]["input_pitch"]
                roll = info["resp"]["input_roll"]

                pitch = np.clip(pitch, -1.0, 1.0)
                roll = np.clip(roll, -1.0, 1.0)
                pitch, roll = int(pitch), int(roll)
                pitch, roll = -pitch, -roll
                action = (-roll, pitch)

        except requests.exceptions.ConnectionError as e:
            print(f"No brain listening on port: {port}", file=sys.stderr)
            raise BrainNotFound
        return action, info

    return next_action_v1


# --------------------------------------------------------------------------------------
def nn_controller(filepath="./ref-no-lstm.npz"):
    npz = np.load(filepath)
    w0 = npz["w0"]
    b0 = npz["b0"]
    w1 = npz["w1"]
    b1 = npz["b1"]
    w_out = npz["w_out"]

    def next_action(state):
        # x = state
        x, y, vel_x, vel_y = state
        x = np.array([x, y, vel_x, vel_y])
        action = w_out @ np.tanh(w1 @ np.tanh(w0 @ x + b0) + b1)

        action = np.clip(action, -1.0, 1.0)
        pitch, roll = action
        return (pitch, roll), {}

    return next_action


# --------------------------------------------------------------------------------------
def pid_controller(Kp=75, Ki=0.5, Kd=45, **kwargs):
    sum_x, sum_y = 0, 0

    def next_action(state):
        nonlocal sum_x, sum_y
        x, y, vel_x, vel_y = state
        sum_x, sum_y = sum_x + x, sum_y + y

        action_x = Kp * x + Ki * sum_x + Kd * vel_x
        action_y = Kp * y + Ki * sum_y + Kd * vel_y
        action_x = np.clip(action_x / 22, -1.0, 1.0)
        action_y = np.clip(action_y / 22, -1.0, 1.0)

        action = (-action_x, -action_y)
        return action, {}

    return next_action


# ======================================================================================
def main(controller, **kwargs):
    controller_fn = controller(**kwargs)
    env = MoabEnv()
    obs = env.reset()
    r_tot = 0

    while True:
        for ep in range(1000):

            for step in range(1, 30):
                action, info = controller_fn(obs)
                obs, reward, done, info = env.step(action)
                r_tot += reward
                time.sleep(0.1)

                env.render()

                if done:
                    break

            obs = env.reset()
            print("Ep len", step, "Ep reward:", r_tot)
            r_tot = 0

    env.close()


if __name__ == "__main__":
    main(nn_controller)
