def mm1(
    state,
    action,
    dt=0.045,  # in s, 45ms
    gravity=9.81,  # m/s^2, Earth: there's no place like it.
    plate_radius=0.225 / 2.0,  # m, Moab: 225mm dia
    ball_mass=0.0027,  # kg, Ping-Pong ball: 2.7g
    ball_radius=0.02,  # m, Ping-Pong ball: 20mm
    ball_shell=0.0002,  # m, Ping-Pong ball: 0.2mm
):
    x_theta, y_theta = action

    # action = np.array([x_theta, y_theta])
    action = np.array([y_theta, -x_theta])

    # Calculate ball intertia with radius and hollow radius
    h = ball_radius - ball_shell  # hollow radius
    r = ball_radius
    g = gravity

    # Ball intertia for a hollow sphere is:
    # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    # Equations for acceleration on a plate at rest
    # a = (theta * m * g) / (m + (I / r**2))

    # acceleration divided by theta (same scaling factor for theta_x and theta_y)
    acc_div_theta = g / (
        1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2)
    )

    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    B = np.array(
        [
            [(1 / 2) * dt**2 * acc_div_theta, 0],
            [0, (1 / 2) * dt**2 * acc_div_theta],
            [dt * acc_div_theta, 0],
            [0, dt * acc_div_theta],
        ]
    )

    state = A @ state + B @ action
    return state


def moab_model(
    state,
    action,
    dt=0.045,
    gravity=9.81,
    ball_mass=0.0027,  # kg, Ping-Pong ball: 2.7g
    ball_radius=0.02,  # m, Ping-Pong ball: 20mm
    ball_shell=0.0002,  # m, Ping-Pong ball: 0.2mm
):
    # action = np.array([y_theta, -x_theta])
    x_theta, y_theta = action
    action = np.array([y_theta, -x_theta])

    # Ball intertia for a hollow sphere is:
    # I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))
    # Equations for acceleration on a plate at rest
    # a = (theta * m * g) / (m + (I / r**2))

    # acceleration divided by theta (same scaling factor for theta_x and theta_y)
    h = ball_radius - ball_shell  # hollow radius
    r = ball_radius
    acc_div_theta = gravity / (
        1 + (2 / 5) * ((r**5 - h**5) / (r**3 - h**3)) / (r**2)
    )

    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    B = np.array(
        [
            [(1 / 2) * dt**2 * acc_div_theta, 0],
            [0, (1 / 2) * dt**2 * acc_div_theta],
            [dt * acc_div_theta, 0],
            [0, dt * acc_div_theta],
        ]
    )

    state = A @ state + B @ action
    return state


def mm2(
    state,
    action,
    dt=0.045,
    gravity=9.81,
    plate_radius=0.225 / 2.0,
    ball_mass=0.0027,
    ball_radius=0.02,
    ball_shell=0.0002,
):
    x_theta, y_theta = action
    theta = np.array([y_theta, -x_theta])

    h = ball_radius - ball_shell
    m = ball_mass
    r = ball_radius
    g = gravity
    I = (2 / 5) * m * ((r**5 - h**5) / (r**3 - h**3))

    ball_acc = (theta * m * g) / (m + (I / r**2))

    ball_disp = state[2:] * dt + (0.5 * ball_acc * (dt**2))
    # print(ball_disp)

    state[:2] += ball_disp
    state[2:] = state[2:] + ball_acc * dt

    return state


x = np.random.randn(4)
u = np.random.randn(2)

print(mm1(x, u))
print(moab_model(x, u))
print(mm2(x, u))
