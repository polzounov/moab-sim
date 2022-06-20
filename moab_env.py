"""
Simulator for the Moab plate+ball balancing device.
"""

import cv2
import pygame
import numpy as np
from time import sleep
from gym import Env, spaces
from moab_sim import MoabSim

pygame.init()


# Controllers ------------------------------------------------------------------
def pid_controller(Kp=3.4, Ki=0.0227, Kd=20.455, **kwargs):
    sum_x, sum_y = 0, 0

    def next_action(state):
        nonlocal sum_x, sum_y
        x, y, vel_x, vel_y = state
        sum_x += x
        sum_y += y

        action_x = Kp * x + Ki * sum_x + Kd * vel_x
        action_y = Kp * y + Ki * sum_y + Kd * vel_y
        action = np.array([-action_x / 22, -action_y / 22])
        return np.clip(action, -1, 1)

    return next_action


# Env --------------------------------------------------------------------------
class MoabEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_iterations=2048):
        act_max = np.asarray([1, 1], dtype=np.float32)
        plate_radius = 0.225 / 2
        obs_max = np.asarray(
            [plate_radius, plate_radius, np.inf, np.inf], dtype=np.float32
        )
        self.observation_space = spaces.Box(-obs_max, obs_max)
        self.action_space = spaces.Box(-act_max, act_max)

        self.iteration_count = 0
        self.max_iterations = max_iterations

        self.sim = MoabSim()
        self.state = self.sim.state
        self._viewer = None

    def close(self):
        pass

    def reset(self):
        self.iteration_count = 0
        self.state = self.sim.reset()
        return self.state

    def done(self) -> bool:
        x, y = self.state[:2]
        halted = np.sqrt(x**2 + y**2) > 0.95 * self.sim.params["plate_radius"]
        halted |= self.iteration_count >= self.max_iterations
        return halted

    def reward(self) -> float:
        x, y, vel_x, vel_y = self.state
        norm_dist = np.sqrt(x**2 + y**2) / self.sim.params["plate_radius"]
        reward = 1 - norm_dist
        return reward

    def step(self, action):
        self.state = self.sim.step(action)
        self.iteration_count += 1
        return self.state, self.reward(), self.done(), {}

    def render(self):
        size = 800
        if self._viewer is None:
            self._viewer = pygame.display.set_mode((size, size))
            self._clock = pygame.time.Clock()

        # Shitty rendering of moab
        x, y, _, _ = self.state
        img = np.zeros((size, size, 3), dtype=np.uint8)
        center = (int(size / 2), int(size / 2))
        scaling = (size / 2) / 0.2  # plate take up halfish of the frame

        # plate
        plate_radius_pix = int(0.1125 * scaling)
        img = cv2.circle(img, center, plate_radius_pix, (100, 100, 100), -1)

        # fmt:off
        # ball
        ball_radius_pix = int(0.020 * scaling)
        ball_x_pix, ball_y_pix = center[0] + int(x * scaling), center[1] - int(y * scaling)
        img = cv2.circle(img, (ball_x_pix, ball_y_pix), ball_radius_pix, (255, 165, 0), -1)
        # fmt: on

        pg_img = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")
        self._viewer.blit(pg_img, (0, 0))
        pygame.display.update()
        # sleep(DELTA_T)
        self._clock.tick(10)

        return img


def main1(Kp=75, Ki=0.5, Kd=45):
    env = MoabEnv()
    state = env.reset()
    controller = pid_controller(Kp, Ki, Kd)
    print(state)

    while True:
        action = controller(state)
        state, reward, done, info = env.step(action)
        env.render()
        print(state, reward, action, done)
        # if done:
        #     break


if __name__ == "__main__":
    pass
    main1(Kp=75, Ki=0.5, Kd=45)
