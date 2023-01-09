import numpy as np
from typing import Dict, Tuple, Optional, Callable


# Moab measured velocity at 15deg in 3/60ths, or 300deg/s
DEFAULT_PLATE_MAX_ANGULAR_VELOCITY = (60.0 / 3.0) * np.radians(15)  # rad/s

# Set acceleration to get the plate up to velocity in 1/100th of a sec
DEFAULT_PLATE_ANGULAR_ACCEL = (
    100.0 / 1.0
) * DEFAULT_PLATE_MAX_ANGULAR_VELOCITY  # rad/s^2


def moab_model(
    state: np.ndarray,
    action: np.ndarray,
    current_dt: float = 1 / 30,
    gravity: float = 9.81,
    ball_radius: float = 0.02,
    ball_shell: float = 0.0002,
    **kwargs,
) -> np.ndarray:
    r = ball_radius
    h = ball_radius - ball_shell  # hollow radius
    dt = current_dt

    # fmt: off
    # Ball intertia for a hollow sphere is:
    # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    # Equations for acceleration on a plate at rest
    # a = (theta * m * g) / (m + (I / r**2))
    # Combine the two to get the acceleration divided by theta
    acc_div_theta = gravity / (1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2))

    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([
            [0.5 * dt**2 * acc_div_theta, 0],
            [0, 0.5 * dt**2 * acc_div_theta],
            [dt * acc_div_theta, 0],
            [0, dt * acc_div_theta],
    ])
    next_state = A @ state + B @ action  # x_t+1 = Ax_t + Bu_t
    # fmt: on
    return next_state


def linear_acceleration(
    acc_magnitude: float,
    max_vel: float,
    dt: float,
    dims: int = 2,
    linalc_type=1,
):
    """
    perform a linear acceleration of a variable towards a destination
    with a hard stop at the destination. returns the position and velocity
    after delta_t has elapsed.

    q:             current position
    dest:          target destination
    vel:           current velocity
    acc_magnitude: acceleration constant
    max_vel:       maximum velocity
    delta_t:       time delta

    returns: (final_position, final_velocity)
    """
    q = np.zeros((dims,))
    vel = np.zeros((dims,))

    assert q.shape == vel.shape

    def next_position(dest: np.ndarray, delta_t: float = dt):
        nonlocal q, vel
        assert q.shape[0] == dims and vel.shape[0] == dims and dest.shape[0] == dims

        # direction of accel
        direc = np.sign(dest - q)

        # calculate the change in velocity and position
        acc = acc_magnitude * direc * delta_t
        vel_end = np.clip(-max_vel, max_vel, vel + acc * delta_t)
        vel_avg = (vel + vel_end) * 0.5
        delta = vel_avg * delta_t
        vel = vel_end

        # Do this for each direction. TODO: do this using vectors...
        for i in range(dims):
            # moving towards the dest?
            if (direc[i] > 0 and q[i] < dest[i] and q[i] + delta[i] < dest[i]) or (
                direc[i] < 0 and q[i] > dest[i] and q[i] + delta[i] > dest[i]
            ):
                q[i] = q[i] + delta[i]

            # stop at dest
            else:
                q[i] = dest[i]
                vel[i] = 0.0

        return q

    return next_position


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
    def __init__(
        self,
        config: Optional[Dict[str, float]] = None,
        linear_acceleration_servos: int = True,
        moab_model_opt: Callable = None,
    ):
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
            "max_starting_distance_ratio": 0.6,  # m, 0.9 of Moab radius
            "max_starting_velocity": 1.0,  # m/s, Ping-Pong ball: 1.0m/s
        }
        if config is not None:
            self._overwrite_params(config)

        self.linear_acceleration_servos = linear_acceleration_servos
        self.lin_acc_fn = linear_acceleration(
            acc_magnitude=DEFAULT_PLATE_ANGULAR_ACCEL,
            max_vel=DEFAULT_PLATE_MAX_ANGULAR_VELOCITY,
            dt=self.params["dt"],
        )

        if moab_model_opt is not None:
            self.moab_model = moab_model_opt
        else:
            self.moab_model = moab_model

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
            # fmt: off
            self.state[:2] = self.params.get("initial_x"), self.params.get("initial_y")
            self.state[2:] = self.params.get("initial_vel_x"), self.params.get("initial_vel_y")
            # fmt: on

        else:
            # Intialize within a uniform circle (uniformly distributed within a circle)
            plate_radius = self.params["plate_radius"]
            max_dist_ratio = self.params["max_starting_distance_ratio"]
            self.state[:2] = uniform_circle(plate_radius * max_dist_ratio)
            self.state[2:] = uniform_circle(self.params["max_starting_velocity"])

        self.lin_acc_fn = linear_acceleration(
            acc_magnitude=DEFAULT_PLATE_ANGULAR_ACCEL,
            max_vel=DEFAULT_PLATE_MAX_ANGULAR_VELOCITY,
            dt=self.params["dt"],
        )

        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Runs a step of simulation.

        action: the target plate angles in radians(!!)
        returns: the next state
        """
        assert action.shape == (2,)

        # Sample jitter and add to dt to get the correct dt for this timestep
        dt = self.params["dt"]
        jitter = self.params["jitter"]
        current_dt = dt + np.random.uniform(-jitter, jitter)

        if self.linear_acceleration_servos:
            self.plate_angles = self.lin_acc_fn(action, delta_t=current_dt)
        else:
            self.plate_angles = action

        self.plate_angles = np.clip(self.plate_angles, -np.radians(22), np.radians(22))

        self.state = self.moab_model(
            self.state, self.plate_angles, current_dt=current_dt, **self.params
        )
        return self.state
