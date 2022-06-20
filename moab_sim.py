import cv2
import numpy as np
from typing import Tuple, Optional


"""
    from scipy.integrate import odeint

    def diff_moab_model(state, action, gravity, ball_radius, ball_shell, **kwargs):
        h = ball_radius - ball_shell  # hollow radius
        r = ball_radius

        # Ball intertia for a hollow sphere is:
        # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
        # Equations for acceleration on a plate at rest
        # a = (theta * m * g) / (m + (I / r**2))
        # Simplify:
        x_dot_dot, y_dot_dot = action * gravity / (
            1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2)
        )
        _, _, x_dot, y_dot = state
        return np.array([x_dot, y_dot, x_dot_dot, y_dot_dot])

    def forward_model_ode(state, action, dt):
    diff_model = lambda s, a: self.diff_moab_model(s, a, **self.params)
    t = np.linspace(0.0, dt, 2)
    return np.array(odeint(self.diff_moab_model, state, t, args=(action, dt)))[1, :]
"""


def moab_model(
    state: np.ndarray,
    action: np.ndarray,
    dt: float = 1 / 30,
    gravity: float = 9.81,
    ball_radius: float = 0.02,
    ball_shell: float = 0.0002,
    **kwargs
) -> np.ndarray:
    # fmt: off
    r = ball_radius
    h = ball_radius - ball_shell  # hollow radius

    # Ball intertia for a hollow sphere is:
    # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    # Equations for acceleration on a plate at rest
    # a = (theta * m * g) / (m + (I / r**2))
    # Combine the two to get the acceleration divided by theta
    acc_div_theta = gravity / (1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2))

    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([
        [(1 / 2) * dt**2 * acc_div_theta, 0],
        [0, (1 / 2) * dt**2 * acc_div_theta],
        [dt * acc_div_theta, 0],
        [0, dt * acc_div_theta]
    ])
    # fmt:on
    next_state = A @ state + B @ action  # x_t+1 = Ax_t + Bu_t
    return next_state


def uniform_circle(r: float) -> Tuple[float, float]:
    """Sample a point uniformly inside a circle with radius r."""
    angle = np.random.uniform(-np.pi, np.pi)

    # If we were to uniformly sample magnitude, points would be concentrated
    # more closer to the center, so we need to sample magnitudes scaled to
    # make everything evenly distributed
    unit_circle_mag = np.random.uniform(0, 1)
    unit_circle_mag = 1 - unit_circle_mag**2  # Basically inverse of r^2
    mag = unit_circle_mag * r

    return mag * np.sin(angle), mag * np.cos(angle)


class MoabSim:
    def __init__(self, config: Optional[dict] = None):
        self.state = np.array([0, 0, 0, 0], dtype=np.float32)

        self.params = {
            "dt": 0.045,  # in s, 45ms
            "gravity": 9.81,  # m/s^2, Earth: there's no place like it.
            "plate_radius": 0.225 / 2,  # m, Moab: 225mm dia
            "ball_mass": 0.0027,  # kg, Ping-Pong ball: 2.7g
            "ball_radius": 0.02,  # m, Ping-Pong ball: 20mm
            "ball_shell": 0.0002,  # m, Ping-Pong ball: 0.2mm
            "max_starting_velocity": 1.0,  # m/s, Ping-Pong ball: 1.0m/s
        }
        if config is not None:
            self._overwrite_params(config)

    def _overwrite_params(config: dict):
        """
        If config exists, overwrite all values of self.params with the matching
        elements in config. (If the element doesn't exist in config, keep the
        one from self.params).
        """
        self.params = self.params | config

    def reset(self, config: Optional[dict] = None) -> np.ndarray:
        if config is not None:
            self._overwrite_params(config)

        self.state[:2] = uniform_circle(0.9 * self.params["plate_radius"])
        self.state[2:] = uniform_circle(self.params["max_starting_velocity"])

        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        self.state = moab_model(self.state, action, **self.params)
        return self.state
