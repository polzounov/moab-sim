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


class MoabEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_iterations=2048, max_plate_angle=22):
        act_max = np.asarray([1, 1], dtype=np.float32)
        plate_radius = 0.1125
        obs_max = np.asarray(
            [plate_radius, plate_radius, np.inf, np.inf], dtype=np.float32
        )
        self.observation_space = spaces.Box(-obs_max, obs_max)
        self.action_space = spaces.Box(-act_max, act_max)

        self.max_plate_angle = max_plate_angle
        self.iteration_count = 0
        self.max_iterations = max_iterations

        self.sim = MoabSim()
        self.state = self.sim.state
        self._viewer = None

    def close(self):
        pass

    def reset(self, config=None):
        self.iteration_count = 0
        self.state = self.sim.reset(config)
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
        # Action is -1, 1 scaled
        action_radians = np.asarray(action) * -1.0 * np.radians(self.max_plate_angle)
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


class MoabDomainRandEnv(MoabEnv):
    def __init__(self, max_iterations=2048, dr_params={}):
        super(MoabDomainRandEnv, self).__init__(max_iterations=max_iterations)

        # defaults for all physics params
        d_g, d_pr, d_bm, d_br, d_bs, d_msv = 9.81, 0.1125, 0.0027, 0.02, 0.0002, 1.0
        self.g = dr_params.get("g") or np.random.uniform(d_g * 0.5, d_g * 2.0)
        self.pr = dr_params.get("pr") or np.random.uniform(d_pr * 0.5, d_pr * 2.0)
        self.bm = dr_params.get("bm") or np.random.uniform(d_bm * 0.5, d_bm * 2.0)
        self.br = dr_params.get("br") or np.random.uniform(d_br * 0.5, d_br * 2.0)
        self.bs = dr_params.get("bs") or np.random.uniform(d_bs * 0.5, d_bs * 2.0)
        self.msv = dr_params.get("msv") or np.random.uniform(d_msv * 0.5, d_msv * 2.0)

        self.dr_config = {
            "g": self.g,
            "pr": self.pr,
            "bm": self.bm,
            "br": self.br,
            "bs": self.bs,
            "msv": self.msv,
        }

    def reset(self):
        self.iteration_count = 0

        # Sample the random physics parameters
        sampled_config = {k: v() for k, v in self.dr_config.items()}

        self.state = self.sim.reset(config=sampled_config)
        return self.state
