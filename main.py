import os
import time
import numpy as np

from functools import partial
from typing import Any, Dict, Union, Optional

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)

from moab_sim import MoabSim
from policies import random_policy, brain_policy

ENABLE_RENDER = False
try:
    # Only import these if using stuff locally
    import cv2
    import pygame

    pygame.init()
    ENABLE_RENDER = True

except:
    # In this case you're probably in a docker container on Azure
    pass


def main(simulator_name, render, max_iterations, workspace=None, accesskey=None):
    # Get workspace and accesskey from env if not passed
    workspace = workspace or os.getenv("SIM_WORKSPACE")
    accesskey = accesskey or os.getenv("SIM_ACCESS_KEY")

    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    registration_info = SimulatorInterface(
        name=simulator_name,
        timeout=60,
        simulator_context=config_client.simulator_context,
        description=None,
    )

    registered_session: SimulatorSessionResponse = client.session.create(
        workspace_name=config_client.workspace,
        body=registration_info,
    )
    print(f"Registered simulator. {registered_session.session_id}")

    sequence_id = 1
    sim_model = MoabBonsaiSim(render=render, max_iterations=max_iterations)
    sim_model_state = sim_model.reset()

    try:
        while True:
            sim_state = SimulatorState(
                sequence_id=sequence_id,
                state=sim_model_state,
                halted=sim_model.done(),
            )
            event = client.session.advance(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
                body=sim_state,
            )
            sequence_id = event.sequence_id

            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
            elif event.type == "EpisodeStart":
                sim_model_state = sim_model.reset(event.episode_start.config)
            elif event.type == "EpisodeStep":
                sim_model_state = sim_model.step(event.episode_step.action)
            elif event.type == "EpisodeFinish":
                pass  # Nothing to do
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because "
                    f"'{event.unregister.details}'"
                )
                return

    except BaseException as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print(f"Unregistered simulator because {type(err).__name__}: {err}")


class MoabBonsaiSim:
    def __init__(
        self,
        render: bool = False,
        max_iterations: int = 2048,
    ):
        """Simulator Interface with the Bonsai Platform

        Parameters
        ----------
        render : bool, optional
            Whether to visualize episodes during training, by default False
            Name of simulator interface, by default "Moab"
        """
        self.simulator = MoabSim()
        self.count_view = False
        self.render = render
        self._viewer = None
        self.iteration_count = 0
        self.max_iterations = max_iterations

    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator."""
        d = self.simulator.params.copy()  # Make a copy
        sim_state = [float(el) for el in self.simulator.state]
        d["ball_x"], d["ball_y"], d["ball_vel_x"], d["ball_vel_y"] = sim_state
        plate_angles = [float(el) for el in self.simulator.plate_angles]
        d["input_pitch"], d["input_roll"] = plate_angles

        return d

    def done(self) -> bool:
        state = self.get_state()
        x, y = state["ball_x"], state["ball_y"]
        out_of_bounds = np.sqrt(x**2 + y**2) > 0.95 * state["plate_radius"]
        too_many_iters = self.iteration_count >= self.max_iterations
        return out_of_bounds or too_many_iters

    def reset(self, config: Dict[str, float] = None) -> Dict[str, float]:
        """Initialize simulator environment using scenario parameters from inkling."""
        self.simulator.reset(config)
        self.iteration_count = 0

        return self.get_state()

    def step(self, action: Dict) -> Dict[str, float]:
        """Step through the environment for a single iteration."""
        # pitch, roll = action["command"]["input_pitch"], action["command"]["input_roll"]
        pitch, roll = action["input_pitch"], action["input_roll"]
        sim_state = self.simulator.step(np.array([pitch, roll], dtype=np.float32))
        self.iteration_count += 1

        if self.render:
            self.render_sim()

        return self.get_state()

    def render_sim(self):
        if ENABLE_RENDER:
            size = 800
            if self._viewer is None:
                self._viewer = pygame.display.set_mode((size, size))
                self._clock = pygame.time.Clock()

            # Shitty rendering of moab
            x, y, _, _ = sim_state = self.simulator.state
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--sim-name",
        type=str,
        metavar="SIMULATOR NAME",
        default="Moab",
        help="Simulator name to use registering with the platform",
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        default=False,
        help="Render training episodes",
    )
    parser.add_argument(
        "--iteration-limit",
        type=int,
        metavar="EPISODE_ITERATIONS",
        help="Episode iteration limit when running local test.",
        default=2048,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        metavar="WORKSPACE ID",
        help="your workspace id",
        default=None,
    )
    parser.add_argument(
        "--accesskey",
        type=str,
        metavar="Your Bonsai workspace access-key",
        help="your bonsai workspace access key",
        default=None,
    )

    args, _ = parser.parse_known_args()
    main(
        simulator_name=args.sim_name,
        render=args.render,
        max_iterations=args.iteration_limit,
        workspace=args.workspace,
        accesskey=args.accesskey,
    )
