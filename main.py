#!/usr/bin/env python3

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2020 Microsoft

Usage:
  For registering simulator with the Bonsai service for training:
    python simulator_integration.py   
    Then connect your registered simulator to a Brain via UI, or using the CLI: `bonsai simulator unmanaged connect -b <brain-name> -a <train-or-assess> -c BalancePole --simulator-name Cartpole
"""

import datetime
import json
import os
import pathlib
import random
import sys
import time
import numpy as np
from typing import Dict, Union, Optional

# from scipy.stats import truncnorm

from dotenv import load_dotenv, set_key
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from functools import partial

from policies import random_policy, brain_policy
from moab_sim import MoabSim

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_PATH = "logs"
default_config = {}


class TemplateSimulatorSession:
    def __init__(
        self,
        render: bool = False,
        env_name: str = "Moab",
        max_iterations: int = 2048,
    ):
        """Simulator Interface with the Bonsai Platform

        Parameters
        ----------
        render : bool, optional
            Whether to visualize episodes during training, by default False
        env_name : str, optional
            Name of simulator interface, by default "Moab"
        """
        self.simulator = MoabSim()
        self.count_view = False
        self.env_name = env_name
        self.render = render

        self.iteration_count = 0
        self.max_iterations = max_iterations

    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator."""
        d = self.simulator.params.copy()  # Make a copy
        sim_state = self.simulator.state
        d["ball_x"], d["ball_y"], d["ball_vel_x"], d["ball_vel_y"] = sim_state
        d["input_pitch"], d["input_roll"] = self.simulator.plate_angles
        d["sim_halted"] = self.halted()
        return d

    def halted(self) -> bool:
        """Halt current episode."""
        x, y, _, _ = self.simulator.state 
        halted = np.sqrt(x**2 + y**2) > 0.95 * self.simulator.params["plate_radius"]
        return halted | self.iteration_count >= self.max_iterations

    def episode_start(self, config: Dict[str, float] = None) -> None:
        """Initialize simulator environment using scenario parameters from inkling."""
        self.simulator.reset(config)
        self.iteration_count = 0

    def episode_step(self, action: Dict):
        """Step through the environment for a single iteration."""
        pitch, roll = action["command"]["input_pitch"], action["command"]["input_roll"]
        self.simulator.step(np.array([pitch, roll], dtype=np.float32))
        self.iteration_count += 1

        if self.render:
            self.sim_render()

    def sim_render(self):
        pass


def ensure_log_dir(log_full_path):
    """
    Ensure the directory for logs exists ??? create if needed.
    """
    print(f"logfile: {log_full_path}")
    logs_directory = pathlib.Path(log_full_path).parent.absolute()
    print(f"Checking {logs_directory}")
    if not pathlib.Path(logs_directory).exists():
        print(
            "Directory does not exist at {0}, creating now...".format(
                str(logs_directory)
            )
        )
        logs_directory.mkdir(parents=True, exist_ok=True)


def env_setup(env_file: str = ".env"):
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(env_file)
    if not env_file_exists:
        open(env_file, "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(env_file, "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(env_file, "SIM_ACCESS_KEY", access_key)

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_policy(
    num_episodes: int = 10,
    render: bool = True,
    num_iterations: int = 2048,
    policy=random_policy,
    policy_name: str = "random",
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = current_time + "_" + policy_name + "_log.csv"
    sim = TemplateSimulatorSession(render=render)

    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        sim_state = sim.episode_start()
        sim_state = sim.get_state()

        while not terminal:
            action = policy(sim_state)
            sim.episode_step(action)
            sim_state = sim.get_state()

            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")

            iteration += 1
            terminal = iteration >= num_iterations

    return sim


def main(
    render: bool = False,
    simulator_name: str = "Moab",
    config_setup: bool = False,
    env_file: Union[str, bool] = ".env",
    workspace: str = None,
    accesskey: str = None,
):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    config_setup: bool, optional
        if enabled then uses a local `.env` file to find sim workspace id and access_key
    sim_speed: int, optional
        the average delay to use, default = 0
    sim_speed_variance: int, optional
        the variance for sim delay
    env_file: str, optional
        if config_setup True, then where the environment variable for lookup exists
    workspace: str, optional
        optional flag from CLI for workspace to override
    accesskey: str, optional
        optional flag from CLI for accesskey to override
    """
    # fmt: off

    # check if workspace or access-key passed in CLI
    use_cli_args = all([workspace, accesskey])

    # use dotenv file if provided
    use_dotenv = env_file or config_setup

    # check for accesskey and workspace id in system variables
    # Three scenarios
    # 1. workspace and accesskey provided by CLI args
    # 2. dotenv provided
    # 3. system variables
    # do 1 if provided, use 2 if provided; ow use 3; if no sys vars or dotenv, fail

    if use_cli_args:
        # BonsaiClientConfig will retrieve as environment variables
        os.environ["SIM_WORKSPACE"] = workspace
        os.environ["SIM_ACCESS_KEY"] = accesskey
    elif use_dotenv:
        if not env_file:
            env_file = ".env"
        print(f"No system variables for workspace-id or access-key found, checking in env-file at {env_file}")
        workspace, accesskey = env_setup(env_file)
        load_dotenv(env_file, verbose=True, override=True)
    else:
        try:
            workspace = os.environ["SIM_WORKSPACE"]
            accesskey = os.environ["SIM_ACCESS_KEY"]
        except:
            raise IndexError(f"Workspace or access key not set or found. Use --config-setup for help setting up.")

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render=render, env_name=simulator_name)

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # # Load json file as simulator integration config type file
    with open("moab_interface.json") as file:
        interface = json.load(file)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )

    def create_session(registration_info: SimulatorInterface, config_client: BonsaiClientConfig):
        """Creates a new Simulator Session and returns new session, sequenceId"""

        try:
            print("config: {}, {}".format(config_client.server, config_client.workspace))
            registered_session: SimulatorSessionResponse = client.session.create(workspace_name=config_client.workspace, body=registration_info)
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print("HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
            raise ex
        except Exception as ex:
            print("UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(ex))
            raise ex

    registered_session, sequence_id = create_session(registration_info, config_client)
    episode = 0
    iteration = 0

    try:
        while True:
            # Proceed to the next event by calling the advance function and passing the simulation state
            # resulting from the previous event. Note that the sim must always be able to return a valid
            # structure from get_state, including the first time advance is called, before an EpisodeStart
            # message has been received.
            sim_state = SimulatorState(
                sequence_id=sequence_id,
                state=sim.get_state(),
                halted=sim.halted(),
            )
            print(sim_state)
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print("[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type))
            except HttpResponseError as ex:
                print("HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = create_session(registration_info, config_client)
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = create_session(registration_info, config_client)
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")

            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1

            elif event.type == "EpisodeStep":
                # Simulate the next state transition using the value of event.episode_step.action.
                # This updates the simulation state, which will be sent back in the next loop when
                # client.session.advance is called.
                iteration += 1
                sim.episode_step(event.episode_step.action)

            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 0

            elif event.type == "Unregister":
                print("Simulator Session unregistered by platform because '{}', Registering again!".format(event.unregister.details))
                registered_session, sequence_id = create_session(registration_info, config_client)
                continue

            else:
                pass

    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        print("Unregistered simulator.")

    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        print("Unregistered simulator because: {}".format(err))
    # fmt: on


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render training episodes",
    )
    parser.add_argument(
        "--sim-name",
        type=str,
        metavar="SIMULATOR NAME",
        default="Moab",
        help="Simulator name to use registering with the platform",
    )
    parser.add_argument(
        "--config-setup",
        action="store_true",
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        metavar="ENVIRONMENT FILE",
        help="path to your environment file",
        default=None,
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-random", action="store_true")
    group.add_argument(
        "--test-exported",
        type=int,
        const=5000,  # if arg is passed with no PORT, use this
        nargs="?",
        metavar="PORT",
        help="Run simulator with an exported brain running on localhost:PORT (default 5000)",
    )
    parser.add_argument(
        "--iteration-limit",
        type=int,
        metavar="EPISODE_ITERATIONS",
        help="Episode iteration limit when running local test.",
        default=2048,
    )

    args, _ = parser.parse_known_args()

    if args.test_random:
        test_policy(render=args.render, policy=random_policy)
    elif args.test_exported:
        port = args.test_exported
        url = f"http://localhost:{port}"
        print(f"Connecting to exported brain running at {url}...")
        trained_brain_policy = partial(brain_policy, exported_brain_url=url)
        test_policy(
            render=args.render,
            policy=trained_brain_policy,
            policy_name="exported",
            num_iterations=args.iteration_limit,
        )
    else:
        main(
            config_setup=args.config_setup,
            simulator_name=args.sim_name,
            render=args.render,
            env_file=args.env_file,
            workspace=args.workspace,
            accesskey=args.accesskey,
        )
