import numpy as np
from typing import Dict, Tuple, Optional, Callable


def moab_model(
    state: np.ndarray,
    action: np.ndarray,
    dt: float = lambda: 1 / 30,
    jitter: float = lambda: 0,
    gravity: float = lambda: 9.81,
    ball_radius: float = lambda: 0.02,
    ball_shell: float = lambda: 0.0002,
    **kwargs,
) -> np.ndarray:
    print("Unused keyword arguments", **kwargs)
    r = ball_radius
    h = ball_radius - ball_shell  # hollow radius
    dt += np.random.uniform(-jitter, jitter)  # add jitter to the simulation timesteps

    # fmt: off
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
            [0, dt * acc_div_theta],
    ])
    next_state = A @ state + B @ action  # x_t+1 = Ax_t + Bu_t
    # fmt: on
    return next_state


def calculate_plate_angles(prev_plate_angles, target_plate_angles):
    """
    Linearly accelerate the real plate angles to the target angles (plate angle
    action output by the contoller).

    TODO: do these calculations, for now we assume it happens instantly.
    """
    return target_plate_angles


def uniform_circle(r: float) -> Tuple[float, float]:
    """Sample a point uniformly inside a circle with radius r."""
    angle = np.random.uniform(-np.pi, np.pi)

    # If we were to uniformly sample magnitude, points would be concentrated
    # more closer to the center, so we need to sample magnitudes scaled to
    # make everything evenly distributed
    unit_circle_mag = np.random.uniform(0, 1)
    unit_circle_mag = 1 - unit_circle_mag**2  # 1 - r^2
    mag = unit_circle_mag * r

    return mag * np.sin(angle), mag * np.cos(angle)


class MoabSim:
    def __init__(self, config: Optional[Dict[str, float]] = None):
        self.state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.plate_angles = np.array([0, 0], dtype=np.float32)

        self.params = {
            "dt": 0.0333,  # in s, 33.3ms
            "jitter": 0.004,  # in s, +/- 4ms
            "gravity": 9.81,  # m/s^2, Earth: there's no place like it.
            "plate_radius": 0.1125,  # m, Moab: 112.5mm radius
            "ball_mass": 0.0027,  # kg, Ping-Pong ball: 2.7g
            "ball_radius": 0.02,  # m, Ping-Pong ball: 20mm
            "ball_shell": 0.0002,  # m, Ping-Pong ball: 0.2mm
            "max_starting_distance_ratio": 0.9,  # m, 0.9 of Moab radius
            "max_starting_velocity": 1.0,  # m/s, Ping-Pong ball: 1.0m/s
        }
        if config is not None:
            self._overwrite_params(config)

    def _overwrite_params(self, config: Dict[str, float]):
        """
        If config exists, overwrite all values of self.params with the matching
        elements in config. (If the element doesn't exist in config, keep the
        one from self.params).
        """
        self.params = self.params | config

    def reset(self, config: Optional[Dict[str, float]] = None) -> np.ndarray:
        if config is not None:
            self._overwrite_params(config)

        if (
            self.params.get("initial_x")
            and self.params.get("initial_y")
            and self.params.get("initial_vel_x")
            and self.params.get("initial_vel_y")
        ):
            # Use the old state intialization (starting x, y, x_vel, and y_vel)
            # (Used in tutorial 2)
            self.state[:2] = self.params.get("initial_x"), self.params.get("initial_y")
            self.state[2:] = self.params.get("initial_vel_x"), self.params.get(
                "initial_vel_y"
            )
        else:
            # Intialize within a uniform circle (uniformly distributed within a circle)
            plate_radius = self.params["plate_radius"]
            max_dist_ratio = self.params["max_starting_distance_ratio"]
            self.state[:2] = uniform_circle(plate_radius * max_dist_ratio)
            self.state[2:] = uniform_circle(self.params["max_starting_velocity"])

        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        # Use the legacy coordinate system for the simulation to match old brains
        # Direction of theta_x and theta_y and scale in degrees
        pitch, roll = -action * 22.0
        action = np.array([-roll, pitch])

        self.plate_angles = calculate_plate_angles(self.plate_angles, action)
        self.state = moab_model(self.state, self.plate_angles, **self.params)
        return self.state
