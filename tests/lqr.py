try:
    import autograd.numpy as np
    from autograd import grad, jacobian
except:
    raise ImportError("Please install autograd.")
try:
    import scipy.linalg as sp_linalg
except:
    raise ImportError("Please install scipy.")


def forward_model(state, action, dt=1 / 300.0):
    x, x_dot, y, y_dot = state
    pitch, roll = action

    x_dot_dot = (3 / 5) * 9.81 * pitch
    y_dot_dot = (3 / 5) * 9.81 * roll

    # Works around a single operating point
    x += x_dot * dt
    x_dot += x_dot_dot * dt
    y += y_dot * dt
    y_dot += y_dot_dot * dt

    # For continuous version of LQR
    # state = np.array([x_dot, x_dot_dot, y_dot, y_dot_dot]).reshape((4,))

    # For discrete version of LQR
    state = np.array([x, x_dot, y, y_dot]).reshape((4,))
    return state


def computeAB(current_state, current_control):
    # Linearizing Dynamics
    forward_dynamics_model = lambda state, action: forward_model(state, action)
    a_mat = jacobian(forward_dynamics_model, 0)
    b_mat = jacobian(forward_dynamics_model, 1)
    A = a_mat(current_state, current_control)
    B = b_mat(current_state, current_control)

    return A, B


def LQR_control():
    # Cost matrices for LQR
    Q = np.diag(np.array([1, 1, 1, 1]))  # state_dimension = 4
    R = np.eye(2)  # control_dimension = 1

    print(Q, R)

    A, B = computeAB(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0]))

    # Use if discrete forward dynamics is used
    X = sp_linalg.solve_discrete_are(A, B, Q, R)
    K = np.dot(np.linalg.pinv(R + np.dot(B.T, np.dot(X, B))), np.dot(B.T, np.dot(X, A)))

    # Use for continuous version of LQR
    # X = sp_linalg.solve_continuous_are(A, B, Q, R)
    # K = np.dot(np.linalg.pinv(R), np.dot(B.T, X))
    return K  # np.squeeze(K, 0)


def main():
    """
    K obtained from dicrete dynamics + discrete LQR and continuous dynamics + continuous LQR should approximately match
    quanser workbook and more importantly achieve balance on the Qube Hardware
    """
    # K_real = [-2.0, 35.0, -1.5, 3.0]  # Correct K from quanser workbook
    K_calc = LQR_control().tolist()
    print("The two following should be close to each other")
    # print("\tThe gains from Quanser are:", K_real)
    print("\tThe calculated gains are:  ", K_calc)


if __name__ == "__main__":
    main()
