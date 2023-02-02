import os
import time
import traceback
import numpy as np

from functools import partial
from typing import Any, Dict, Union, Optional, Callable

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)

from moab_sim import MoabSim


def main():
    # Get workspace and accesskey from env if not passed
    workspace = os.getenv("SIM_WORKSPACE")
    accesskey = os.getenv("SIM_ACCESS_KEY")

    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    registration_info = SimulatorInterface(
        name="Moab-py",
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
    sim_model = MoabBonsaiSim()
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
        traceback.print_exc()


class MoabBonsaiSim:
    def __init__(
        self,
        max_iterations: int = 300,
        linear_acceleration_servos: bool = True,
        quantize: bool = False,
        moab_model_opt: Optional[Callable] = None,
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
        self._viewer = None
        self.iteration_count = 0
        self.max_iterations = max_iterations

    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator."""
        d = self.simulator.params.copy()  # Make a copy

        x, y, vel_x, vel_y = self.simulator.state
        pitch, roll = self.simulator.plate_angles

        d["ball_x"], d["ball_y"] = x, y
        d["ball_vel_x"], d["ball_vel_y"] = vel_x, vel_y
        d["pitch"], d["roll"] = pitch, roll

        # Calculate the plate normal. TODO: double check the math
        # This is ONLY for the visualizer, not for correctness of physics
        d["plate_nor_x"] = float(np.cos(np.radians(pitch)) * np.sin(np.radians(roll)))
        d["plate_nor_y"] = float(np.sin(np.radians(pitch)) * np.cos(np.radians(roll)))
        d["plate_nor_z"] = float(np.cos(np.radians(pitch)) * np.cos(np.radians(roll)))

        return d

    def reset(self, config: Dict[str, float] = None) -> Dict[str, float]:
        """Initialize simulator environment using scenario parameters from inkling."""
        self.simulator.reset(config)
        self.iteration_count = 0
        return self.get_state()

    def done(self) -> bool:
        """The terminal function, detects if ball is off the plate."""
        state = self.get_state()
        x, y = state["ball_x"], state["ball_y"]
        halted = np.sqrt(x**2 + y**2) > state["plate_radius"]
        halted |= self.iteration_count >= self.max_iterations
        return halted

    def step(self, action: Dict) -> Dict[str, float]:
        """Step through the environment for a single iteration."""
        pitch, roll = action["input_pitch"], action["input_roll"]
        action = np.array([pitch, roll], dtype=np.float32) * np.radians(22)
        pitch, roll = np.clip(-1, 1, action)

        # In order to maintain brain compatibility on the hardware with brains
        # trained on a previous version of the simulator
        action_legacy = (roll, -pitch)

        # Run the sim!
        self.simulator.step(action_legacy)

        self.iteration_count += 1
        return self.get_state()


if __name__ == "__main__":
    main()
