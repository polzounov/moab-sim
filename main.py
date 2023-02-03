import os
import math
import time
import traceback

from typing import Dict, Optional, Callable

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)

from moab_sim import MoabSim, clip


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
    def __init__(self, max_iterations: int = 300):
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

        # Ensure that everything is a float since Bonsai doesn't support numpy
        # floats, they MUST be python floats
        d["ball_x"], d["ball_y"] = float(x), float(y)
        d["ball_vel_x"], d["ball_vel_y"] = float(vel_x), float(vel_y)
        d["pitch"], d["roll"] = float(pitch), float(roll)

        # Calculate the plate normal. TODO: double check the math
        # This is ONLY for the visualizer, not for correctness of physics
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        d["plate_nor_x"] = float(math.cos(pitch_rad) * math.sin(roll_rad))
        d["plate_nor_y"] = float(math.sin(pitch_rad) * math.cos(roll_rad))
        d["plate_nor_z"] = float(math.cos(pitch_rad) * math.cos(roll_rad))

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
        halted = math.sqrt(x**2 + y**2) > state["plate_radius"]
        halted |= self.iteration_count >= self.max_iterations
        return halted

    def step(self, action: Dict) -> Dict[str, float]:
        """Step through the environment for a single iteration."""
        pitch, roll = action["input_pitch"], action["input_roll"]
        pitch, roll = clip((pitch, roll), -1, 1)

        # In order to maintain brain compatibility on the hardware with brains
        # trained on a previous version of the simulator
        action_legacy = (roll, -pitch)

        # Run the sim!
        self.simulator.step(action_legacy)

        self.iteration_count += 1
        return self.get_state()


if __name__ == "__main__":
    main()
