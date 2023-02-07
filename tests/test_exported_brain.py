import os
import sys
import time
import requests
import numpy as np
import logging as log
from argparse import ArgumentParser

sys.path.append(os.getcwd() + "/../Baselines")
from moab_env import MoabEnv


def brain_controller(
    max_angle=22,
    port=5555,
    client_id=123,
    **kwargs,
):
    """
    This class interfaces with an HTTP server running locally.
    It passes the current hardware state and gets new plate
    angles in return.
    The hardware state is unprojected from camera pixel space
    back to real space by using the calculated plate surface plane.

    Note: Still use v1 endpoint even if it's a v2 brain because on every new
    creation of a brain controller we still call DELETE on the v2 brain
    endpoint. This way we don't need to know information about what the trained
    brain was called to navigate the json response.
    """
    # Reset memory if a v2 brain
    status = requests.delete(f"http://localhost:{port}/v2/clients/{client_id}")
    version = 2 if status.status_code == 204 else 1

    if version == 1:
        prediction_url = f"http://localhost:{port}/v1/prediction"
    elif version == 2:
        prediction_url = f"http://localhost:{port}/v2/clients/{client_id}/predict"
    else:
        raise ValueError("Brain version `{self.version}` is not supported.")

    def next_action_v1(state):
        x, y, vel_x, vel_y = state
        x, y, vel_x, vel_y = float(x), float(y), float(vel_x), float(vel_y)

        observables = {
            "ball_x": x,
            "ball_y": y,
            "ball_vel_x": vel_x,
            "ball_vel_y": vel_y,
        }

        action = (0, 0)  # Action is 0,0 if not detected or brain didn't work
        info = {"status": 400, "resp": ""}
        # Trap on GET failures so we can restart the brain without
        # bringing down this run loop. Plate will default to level
        # when it loses the connection.
        try:
            # Get action from brain
            response = requests.get(prediction_url, json=observables)
            info = {"status": response.status_code, "resp": response.json()}

            if response.ok:
                pitch = info["resp"]["input_pitch"]
                roll = info["resp"]["input_roll"]

                # Scale and clip
                pitch = np.clip(pitch * max_angle, -max_angle, max_angle)
                roll = np.clip(roll * max_angle, -max_angle, max_angle)

                # To match how the old brain works (only integer plate angles)
                action = (-roll, pitch)

        except requests.exceptions.ConnectionError as e:
            print(f"No brain listening on port: {port}", file=sys.stderr)
            raise BrainNotFound
        except Exception as e:
            print(f"Brain exception: {e}")
        return action, info

    def next_action_v2(state):
        x, y, vel_x, vel_y = state
        x, y, vel_x, vel_y = float(x), float(y), float(vel_x), float(vel_y)

        observables = {
            "state": {
                "ball_x": x,
                "ball_y": y,
                "ball_vel_x": vel_x,
                "ball_vel_y": vel_y,
            }
        }

        action = (0, 0)  # Action is 0,0 if not detected or brain didn't work
        info = {"status": 400, "resp": ""}
        # Trap on GET failures so we can restart the brain without
        # bringing down this run loop. Plate will default to level
        # when it loses the connection.
        try:
            # Get action from brain
            response = requests.post(prediction_url, json=observables)
            info = {"status": response.status_code, "resp": response.json()}

            if response.ok:
                concepts = info["resp"]["concepts"]
                concept_name = list(concepts.keys())[0]  # Just use first concept
                pitch = concepts[concept_name]["action"]["input_pitch"]
                roll = concepts[concept_name]["action"]["input_roll"]

                # Scale and clip
                pitch = np.clip(pitch * max_angle, -max_angle, max_angle)
                roll = np.clip(roll * max_angle, -max_angle, max_angle)

                # To match how the old brain works (only integer plate angles)
                action = (-roll, pitch)

        except requests.exceptions.ConnectionError as e:
            print(f"No brain listening on port: {port}", file=sys.stderr)
            raise BrainNotFound
        except Exception as e:
            print(f"Brain exception: {e}")
        return action, info

    if version == 1:
        return next_action_v1
    elif version == 2:
        return next_action_v2
    else:
        raise ValueError("Brain version `{self.version}` is not supported.")


def main(port=5555):
    brain = brain_controller(port=port)
    env = MoabEnv()

    while True:
        done = False
        state = env.reset()
        while not done:
            action, info = brain(state)
            state, _, done, _ = env.step(np.asarray(action))
            time.sleep(1 / 30)
            env.render()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    main(port=args.port)
