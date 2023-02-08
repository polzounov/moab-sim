"""
Simulator for the Moab plate+ball balancing device.
"""

import os
import sys
import cv2
import pygame
import numpy as np
from time import sleep
from gym import Env, spaces

sys.path.append(os.getcwd() + "/..")
from moab_sim import MoabSim

pygame.init()


class MoabEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_iterations=300,
        linear_acceleration_servos=True,
        quantize=False,
        camera_adjust=False,
    ):
        self.sim = MoabSim(
            linear_acceleration_servos=linear_acceleration_servos,
            quantize=quantize,
            # camera_adjust=camera_adjust,
        )

        act_max = np.asarray([1, 1], dtype=np.float32)
        plate_radius = self.sim.params["plate_radius"]
        obs_max = np.asarray(
            [plate_radius, plate_radius, np.inf, np.inf], dtype=np.float32
        )
        self.observation_space = spaces.Box(-obs_max, obs_max)
        self.action_space = spaces.Box(-act_max, act_max)
        self.max_iterations = max_iterations
        self.state = self.sim.state
        self.iteration_count = 0
        self._viewer = None

    def close(self):
        pass

    def reset(self, config=None):
        state = self.sim.reset(config)
        self.iteration_count = 0

        self.state = np.asarray(state, dtype=np.float32)
        return self.state

    def done(self) -> bool:
        x, y = self.state[:2]
        halted = np.sqrt(x**2 + y**2) > self.sim.params["plate_radius"]
        halted |= self.iteration_count >= self.max_iterations
        return halted

    def reward(self) -> float:
        x, y, vel_x, vel_y = self.state
        norm_dist = np.sqrt(x**2 + y**2) / self.sim.params["plate_radius"]
        reward = 1 - norm_dist
        return reward

    def step(self, action):
        action = np.asarray(action)  # Action is -1, 1 scaled
        pitch, roll = np.clip(-1, 1, action)
        pitch, roll = float(pitch), float(roll)

        # Legacy action mapping (for compatibility with old brains)
        action_legacy = (roll, -pitch)

        state = self.sim.step(action_legacy)
        self.state = np.asarray(state, dtype=np.float32)

        self.iteration_count += 1
        return self.state, self.reward(), self.done(), {}

    def render(self, mode=None):
        size = 800
        if self._viewer is None:
            self._viewer = pygame.display.set_mode((size, size))
            self._clock = pygame.time.Clock()

        x, y, _, _ = self.state
        img = np.zeros((size, size, 3), dtype=np.uint8)
        center = (int(size / 2), int(size / 2))
        scaling = (size / 2) / 0.2  # plate take up halfish of the frame

        # plate q
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


class MoabPartialDomainRandEnv(MoabEnv):
    def reset(self):
        self.iteration_count = 0

        # Default values of physical constants being randomized
        d_b_radius, d_b_shell = 0.02, 0.0002
        br = np.random.uniform(d_b_radius * 0.8, d_b_radius * 1.2)
        bs = np.random.uniform(d_b_shell * 0.8, d_b_shell * 1.2)

        config = {
            "ball_radius": br,
            "ball_shell": bs,
        }

        self.state = self.sim.reset(config=config)
        return self.state


class MoabDomainRandEnv(MoabEnv):
    def reset(self):
        self.iteration_count = 0

        # Default values of physical constants being randomized
        d_gravity, d_p_radius, d_b_radius, d_b_shell = 9.81, 0.1125, 0.02, 0.0002
        g = np.random.uniform(d_gravity * 0.8, d_gravity * 1.2)
        pr = np.random.uniform(d_p_radius * 0.8, d_p_radius * 1.2)
        br = np.random.uniform(d_b_radius * 0.8, d_b_radius * 1.2)
        bs = np.random.uniform(d_b_shell * 0.8, d_b_shell * 1.2)

        config = {
            "gravity": g,
            "plate_radius": pr,
            "ball_radius": br,
            "ball_shell": bs,
        }

        self.state = self.sim.reset(config=config)
        return self.state


class MoabDomainRandEnv2(MoabEnv):
    def reset(self):
        self.iteration_count = 0

        # Default values of physical constants being randomized
        d_gravity, d_p_radius, d_b_radius, d_b_shell = 9.81, 0.1125, 0.02, 0.0002
        g = np.random.uniform(d_gravity * 0.8, d_gravity * 1.2)
        pr = np.random.uniform(d_p_radius * 0.8, d_p_radius * 1.2)
        br = np.random.uniform(d_b_radius * 0.8, d_b_radius * 1.2)
        bs = np.random.uniform(d_b_shell * 0.8, d_b_shell * 1.2)

        config = {
            "gravity": g,
            "plate_radius": pr,
            "ball_radius": br,
            "ball_shell": bs,
        }

        self.state = self.sim.reset(config=config)
        return self.state
