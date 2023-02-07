import numpy as np
from typing import Dict


def pid_controller(Kp=3.4, Ki=0.0227, Kd=20.455, **kwargs):
    sum_x, sum_y = 0, 0

    def next_action(state: Dict[str, float]):
        nonlocal sum_x, sum_y
        print(state)
        x, y, vel_x, vel_y = (
            state["ball_x"],
            state["ball_y"],
            state["ball_vel_x"],
            state["ball_vel_y"],
        )
        sum_x += x
        sum_y += y

        action_x = Kp * x + Ki * sum_x + Kd * vel_x
        action_y = Kp * y + Ki * sum_y + Kd * vel_y
        action = np.array([-action_x / 22, -action_y / 22])
        # return np.clip(action, -1, 1)
        pitch, roll = np.clip(action, -1, 1)
        return {"command": {"input_pitch": pitch, "input_roll": roll}}

    return next_action


def random_policy(state: Dict[str, float], **kwargs):
    pitch, roll = np.clip(np.random.randn(2), -1, 1)
    return {"command": {"input_pitch": pitch, "input_roll": roll}}


def brain_policy(
    state: Dict[str, float],
    exported_brain_url: str = "http://localhost:5000",
    **kwargs,
):

    prediction_endpoint = f"{exported_brain_url}/v1/prediction"
    response = requests.get(prediction_endpoint, json=state)

    return response.json()
