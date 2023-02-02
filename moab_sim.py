import math
import random
import numpy as np
from typing import Dict, Tuple, Optional, Callable


# Moab measured velocity at 15deg in 3/60ths, or 300deg/s
DEFAULT_PLATE_MAX_ANGULAR_VELOCITY = (60.0 / 3.0) * math.radians(15)  # rad/s

# Set acceleration to get the plate up to velocity in 1/100th of a sec
DEFAULT_PLATE_ANGULAR_ACCEL = (100.0 / 1.0) * DEFAULT_PLATE_MAX_ANGULAR_VELOCITY  # rad/s^2

# The maximum angle we allow of the Moab plate
MAX_PLATE_ANGLE = math.radians(22)  # rad


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp a value, tuple, or array between a min and max."""
    if isinstance(val, (float, int)):
        return min(max_val, max(min_val, val))
    elif isinstance(val, (tuple)):
        return tuple([clamp(v, min_val, max_val) for v in val])
    elif isinstance(val, (list)):
        return [clamp(v, min_val, max_val) for v in val]
    else:
        raise TypeError("val must be a float, int, tuple, or list")


def moab_model(
    state: Tuple[float, float, float, float],
    action: Tuple[float, float],
    current_dt: float,
    gravity: float,
    ball_radius: float,
    ball_shell: float,
    **kwargs,
) -> Tuple[float, float, float, float]:
    """
    Runs a single step of the moab simulation.

    Make the 3d problem into a 2d problem. Since the plate is a 2d plane and
    the ball moves due to gravity, we can represent the ball's acceleration as
    a function of the plate's tilt (pitch and roll). We can then use the
    equations of motion to calculate the ball's next position and velocity.

    We can calculate the acceleration of the ball as:
        a = (theta * m * g) / (m + (I / r**2))
    Where theta is the plate's tilt along the x or y direction, m is the mass,
    g is gravity, and I is the moment of inertia of the ball.

    And the moment of inertia of a hollow sphere is:
        I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    Where r is the radius of the ball and h is the radius of the hollow part.


    Then we can use the equations of motion to calculate the ball's next
    position and velocity:
        displacement = prev_velocity * dt + 1/2 * acceleration * dt^2
        velocity = prev_velocity + acceleration * dt


    args:
        state:          (x, y, vel_x, vel_y) in (m, m, m/s, m/s)
        action:         (pitch, roll) in (rad, rad)
        current_dt:     time delta in seconds
        gravity:        gravity constant in m/s^2
        ball_radius:    radius of the ball in m
        ball_shell:     radius of the hollow part of the ball in m

    returns: next_state: (x, y, vel_x, vel_y)
    """
    x, y, vel_x, vel_y = state
    pitch, roll = action  # pitch is along x (theta_x), roll is along y (theta_y)

    r = ball_radius
    h = ball_radius - ball_shell  # hollow radius
    dt = current_dt

    # Ball intertia for a hollow sphere is:
    # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    # Equations for acceleration on a plate at rest
    # a = (theta * m * g) / (m + (I / r**2))
    # Combine the two to get the acceleration divided by theta
    acc_div_theta = gravity / (1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2))

    # Equations of motion:
    x += vel_x * dt + 1 / 2 * (acc_div_theta * pitch) * dt**2
    vel_x += (acc_div_theta * pitch) * dt

    y += vel_y * dt + 1 / 2 * (acc_div_theta * roll) * dt**2
    vel_y += (acc_div_theta * roll) * dt

    return x, y, vel_x, vel_y


def linear_acceleration(
    acc_magnitude: float,
    max_vel: float,
    dt: float,
    dims: int = 2,
):
    """
    Perform a linear acceleration of a variable towards a destination
    with a hard stop at the destination. returns the position and velocity
    after current_dt has elapsed.

    args:
        q:             current position
        dest:          target destination
        vel:           current velocity
        acc_magnitude: acceleration constant
        max_vel:       maximum velocity
        current_dt:    time delta

    returns: (final_position, final_velocity)
    """
    q = np.zeros((dims,))
    vel = np.zeros((dims,))

    assert q.shape == vel.shape

    def next_position(dest: np.ndarray, current_dt: float = dt):
        nonlocal q, vel
        assert q.shape[0] == dims and vel.shape[0] == dims and dest.shape[0] == dims

        # direction of accel
        direc = np.sign(dest - q)

        # calculate the change in velocity and position
        acc = acc_magnitude * direc * current_dt
        vel_end = np.clip(-max_vel, max_vel, vel + acc * current_dt)
        vel_avg = (vel + vel_end) * 0.5
        delta = vel_avg * current_dt
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
    """
    Sample a point uniformly inside a circle with radius r.

    args:
        r: radius of the circle

    returns: sampled_point (x, y)
    """
    angle = random.uniform(-math.pi, math.pi)

    # If we were to uniformly sample magnitude, points would be concentrated
    # more closer to the center, so we need to sample magnitudes scaled to
    # make everything evenly distributed
    unit_circle_mag = random.uniform(0, 1)
    unit_circle_mag = 1 - unit_circle_mag**2  # 1 - r^2
    mag = unit_circle_mag * r

    return mag * math.sin(angle), mag * math.cos(angle)


class MoabSim:
    def __init__(
        self,
        config: Optional[Dict[str, float]] = None,
        linear_acceleration_servos: bool = True,
        camera_adjust: bool = True,
    ):
        """
        args:
            config: dictionary of parameters to override the default physics
            linear_acceleration_servos: use linear acceleration for the servos
        """
        self.state = (0, 0, 0, 0)
        self.plate_angles = (0, 0)

        self.params = {
            "dt": 0.0333,  # in s, 33.3ms
            "jitter": 0.004,  # in s, +/- 4ms
            "gravity": 9.81,  # m/s^2, Earth: there's no place like it.
            "plate_radius": 0.1125,  # m, Moab: 112.5mm radius
            "ball_mass": 0.0027,  # kg, Ping-Pong ball: 2.7g
            "ball_radius": 0.02,  # m, Ping-Pong ball: 20mm
            "ball_shell": 0.0002,  # m, Ping-Pong ball: 0.2mm
            "max_starting_distance_ratio": 0.6,  # m, 0.6 of Moab plate radius
            "max_starting_velocity": 1.0,  # m/s, Ping-Pong ball: 1.0m/s
        }
        if config is not None:
            self._overwrite_params(config)

        # TODO: explain what this does
        self.use_linear_acceleration_servos = linear_acceleration_servos
        self.lin_acc_fn = linear_acceleration(
            acc_magnitude=DEFAULT_PLATE_ANGULAR_ACCEL,
            max_vel=DEFAULT_PLATE_MAX_ANGULAR_VELOCITY,
            dt=self.params["dt"],
        )

    def _overwrite_params(self, config: Dict[str, float]):
        """
        If config exists, overwrite all values of self.params with the matching
        elements in config. (If the element doesn't exist in config, keep the
        one from self.params). This is used to override the default physics
        parameters with the ones specified in the config, this allows us to
        easily change the physics parameters through inkling by passing config.
        """
        self.params = self.params | config

    def reset(self, config: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
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

            # The downside of this is that the ball starting points are not
            # uniformly distributed within the circle, but rather uniformly
            # distributed within a square on the plate. (ie this is not
            # symmetric around the plate)
            x = self.params.get("initial_x")
            y = self.params.get("initial_y")
            x_vel = self.params.get("initial_vel_x")
            y_vel = self.params.get("initial_vel_y")
            self.state = (x, y, x_vel, y_vel)

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

    def step(self, action: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """
        Runs a step of simulation.

        action: the target plate angles in radians(!!)
        returns: the next state
        """

        # Sample jitter and add to dt to get the correct dt for this timestep
        # (we need to sample here to also pass the correct dt to the linear
        # acceleration function)
        dt = self.params["dt"]
        jitter = self.params["jitter"]
        current_dt = dt + random.uniform(-jitter, jitter)

        # By having the linear acceleration servos as optional, we can test how
        # the linear acceleration of the servos will impact the sim2real
        # transfer (by adding additional simulation fidelity)
        if self.use_linear_acceleration_servos:
            plate_angles = self.lin_acc_fn(
                np.asarray(action, dtype=np.float32), current_dt=current_dt
            )
            self.plate_angles = tuple(plate_angles)
        else:
            self.plate_angles = action

        # Limit the plate angles to 22 degrees (same as in hardware)
        self.plate_angles = clamp(self.plate_angles, -math.radians(22), math.radians(22))

        self.state = moab_model(self.state, self.plate_angles, current_dt=current_dt, **self.params)

        return self.state
