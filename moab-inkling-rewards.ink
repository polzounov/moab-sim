###

# MSFT Bonsai 
# Copyright 2020 Microsoft
# This code is licensed under MIT license (see LICENSE for details)

# Moab Tutorial 1
# This introductory sample demonstrates how to teach a policy for 
# controlling a ball on the plate of a "Moab" hardware device. 

# To understand this Inkling better, please follow our tutorial walkthrough: 
# https://aka.ms/moab/tutorial1

###

inkling "2.0"

using Math
using Goal

# Distances measured in meters
const RadiusOfPlate = 0.1125 # m

# Velocities measured in meters per sec.
const MaxVelocity = 6.0
const MaxInitialVelocity = 1.0

# Threshold for ball placement
const CloseEnough = 0.02

# Ping-Pong ball constants
const PingPongRadius = 0.020 # m
const PingPongShell = 0.0002 # m

# Default time delta between simulation steps (s)
const DefaultTimeDelta = 0.045

# Maximum distance per step in meters
const MaxDistancePerStep = DefaultTimeDelta * MaxVelocity

# State received from the simulator after each iteration
type ObservableState {
    # Ball X,Y position
    ball_x: number<-MaxDistancePerStep - RadiusOfPlate .. RadiusOfPlate + MaxDistancePerStep>,
    ball_y: number<-MaxDistancePerStep - RadiusOfPlate .. RadiusOfPlate + MaxDistancePerStep>,

    # Ball X,Y velocity
    ball_vel_x: number<-MaxVelocity .. MaxVelocity>,
    ball_vel_y: number<-MaxVelocity .. MaxVelocity>,
}

# Action provided as output by policy and sent as
# input to the simulator
type SimAction {
    # Range -1 to 1 is a scaled value that represents
    # the full plate rotation range supported by the hardware.
    input_pitch: number<-1 .. 1>, # rotate about x-axis
    input_roll: number<-1 .. 1>, # rotate about y-axis
}

# Per-episode configuration that can be sent to the simulator.
# All iterations within an episode will use the same configuration.
type SimConfig {
    # Model initial ball conditions
    initial_x: number<-RadiusOfPlate .. RadiusOfPlate>, # in (m)
    initial_y: number<-RadiusOfPlate .. RadiusOfPlate>,

    # Model initial ball velocity conditions
    initial_vel_x: number<-MaxInitialVelocity .. MaxInitialVelocity>, # in (m/s)
    initial_vel_y: number<-MaxInitialVelocity .. MaxInitialVelocity>,

    # Range -1 to 1 is a scaled value that represents
    # the full plate rotation range supported by the hardware.
    initial_pitch: number<-1 .. 1>,
    initial_roll: number<-1 .. 1>,

    ball_radius: number, # Radius of the ball in (m)
    ball_shell: number, # Shell thickness of ball in (m), shell>0, shell<=radius

}


function RewardFn(State: SimState): number {  
    var reward = 1 - (Math.sqrt(State.ball_x*State.ball_x + State.ball_y*State.ball_y) / RadiusOfPlate)
    if (reward < 0.9) {
        return reward
    } else {
        return reward + 1
    }
}


# Define a concept graph with a single concept
graph (input: ObservableState) {
    concept MoveToCenter(input): SimAction {
        curriculum {
            # The source of training for this concept is a simulator that
            #  - can be configured for each episode using fields defined in SimConfig,
            #  - accepts per-iteration actions defined in SimAction, and
            #  - outputs states with the fields defined in SimState.
            source simulator MoabSim(Action: SimAction, Config: SimConfig): ObservableState {
                # Automatically launch the simulator with this
                # registered package name.
                package "Moab"
            }

            reward RewardFn

            training {
                # Limit episodes to 250 iterations instead of the default 1000.
                EpisodeIterationLimit: 250
            }

            lesson `Randomize Start` {
                # Specify the configuration parameters that should be varied
                # from one episode to the next during this lesson.
                scenario {
                    initial_x: number<-RadiusOfPlate * 0.6 .. RadiusOfPlate * 0.6>,
                    initial_y: number<-RadiusOfPlate * 0.6 .. RadiusOfPlate * 0.6>,

                    initial_vel_x: number<-MaxInitialVelocity * 0.4 .. MaxInitialVelocity * 0.4>,
                    initial_vel_y: number<-MaxInitialVelocity * 0.4 .. MaxInitialVelocity * 0.4>,

                    initial_pitch: number<-0.2 .. 0.2>,
                    initial_roll: number<-0.2 .. 0.2>,

                    ball_radius: number<PingPongRadius * 0.8 .. PingPongRadius * 1.2>,
                    ball_shell: number<PingPongShell * 0.8 .. PingPongShell * 1.2>,
                }
            }
        }
    }
}

# Special string to hook up the simulator visualizer
# in the web interface.
const SimulatorVisualizer = "/moabviz/"
