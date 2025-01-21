import asyncio
import time

try:
    import ujson as json
except ModuleNotFoundError:
    print(
        "Warning: ujson library not found, using native json which is much slower and could cause issues."
    )
    import json

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

from score import (
    compute_energy_of_ref_file,
    compute_angles,
    are_angles_close,
    majority_voting,
)
from skeleton_utils import normalize_skeleton, JOINTS_NAMES_TO_IDX
from utils import (
    extract_ref_motion_data,
    convert_angles_names_to_idx,
    get_orthogonal_indices,
)


# Which axes are front up etc CHANGES for the incoming data depending on initialization
SPECTATOR_XY_AXES = None

# This should be fixed!
REFERENCE_XY_AXES = [0, 1]

# DIFFICULTY SETTINGS
ANGLES_TOLERANCE = 30  # in degrees

# OSC
OSC_PORT = 8080  # PYTHON SERVER PORT
OSC_CLIENT_PORT = 9001  # UNREAL PORT
OSC_IP = "127.0.0.1"
RESULT_INTERVAL = 1  # Interval at which to send results of the analysis (in seconds)

# PARAMETERS
USE_3D = False  # TODO
MIRROR = False

# SCORE COMPUTATION
JOINTS_USED_FOR_ENERGY = ["LEFT_HAND_MIDDLE_4", "RIGHT_HAND_MIDDLE_4"]
ANGLES_USED_FOR_SCORE = convert_angles_names_to_idx(
    [  # should be tuples of 3 keypoints, the one in the middle being the one with the
        ("LeftShoulder", "LeftArm", "LeftForeArm"),  # Left shoulder
        ("LeftArm", "LeftForeArm", "LeftHand"),  # Left elbow
        ("RightShoulder", "RightArm", "RightForeArm"),  # Right shoulder
        ("RightArm", "RightForeArm", "RightHand"),  # Right elbow
    ]
)

# DATA ABOUT CURRENT LEVEL
CURRENT_LEVEL = "choreography"
ENERGY_ARRAY = None
REF_MOTION = None
REF_FRAMETIME = 1 / 30  # It should not change!!!!!
REF_ENERGY = None
REF_ANGLES = None
REF_MOTION = None

IDX_TO_LEVEL = {0: "choreography", 1: "mutation_dance"}

LEVEL_TO_IDX = {
    "choreography": 0,
    "mutation_dance": 1,
}

# History to compute energy and smoothing
previous_spec_frame = None
prev_left_hand_velocity = None
prev_right_hand_velocity = None
# Smoothing
spectator_angles_history = []  # To store spectator angles over RESULT_INTERVAL
left_hand_energy_history = []  # To store left hand energy over RESULT_INTERVAL
right_hand_energy_history = []  # To store right hand energy over RESULT_INTERVAL
last_send_time = time.time()


# Client to senc OSC messages back to Unreal
client = SimpleUDPClient(OSC_IP, OSC_CLIENT_PORT)  # Create an OSC client


def answer_ping(address, *args):
    print("Received ping!")
    client.send_message("/answer", json.dumps("pong"))


def load_level(address, *args):
    global CURRENT_LEVEL
    CURRENT_LEVEL = IDX_TO_LEVEL(args[0])
    print(f"Changed level to {IDX_TO_LEVEL(args[0])}")


def process_and_send_data(address, *args):
    """Receive data via OSC, compute metrics, and send smoothed results back."""
    global SPECTATOR_XY_AXES
    global previous_spec_frame, prev_left_hand_velocity, prev_right_hand_velocity
    global spectator_angles_history, left_hand_energy_history, right_hand_energy_history, last_send_time

    try:
        # Parse incoming OSC message
        timestamp = args[-1]  # in ms
        ref_frame_idx = int(timestamp / REF_FRAMETIME) % len(REF_MOTION)
        ref_frame = REF_MOTION["choreography"][ref_frame_idx][:, REFERENCE_XY_AXES]

        # Reshape incoming frame and normalize skeleton
        raw_spectator_frame = np.array(args[:-4]).reshape(-1, 3)
        spectator_frame = normalize_skeleton(raw_spectator_frame)
        if SPECTATOR_XY_AXES is None:
            SPECTATOR_XY_AXES = get_orthogonal_indices(
                spectator_frame[JOINTS_NAMES_TO_IDX["LeftShoulder"]],
                spectator_frame[JOINTS_NAMES_TO_IDX["RightShoulder"]],
                spectator_frame[JOINTS_NAMES_TO_IDX["Neck"]],
                spectator_frame[JOINTS_NAMES_TO_IDX["Hips"]],
            )
        spectator_frame = spectator_frame[:, SPECTATOR_XY_AXES]  # Work in 2D

        # Compute angles for the current spectator frame
        spectator_angles = compute_angles(
            spectator_frame[np.newaxis, :], ANGLES_USED_FOR_SCORE
        )[0]

        # Store angles for smoothing
        spectator_angles_history.append(spectator_angles)

        # Initialize previous frame and velocities on the first frame
        if previous_spec_frame is None:
            previous_spec_frame = spectator_frame
            prev_left_hand_velocity = np.zeros(2)  # Assuming 2D
            prev_right_hand_velocity = np.zeros(2)  # Assuming 2D
            return  # Skip further processing for the first frame

        # Compute velocities for hands
        left_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
        right_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["RightHand"]]

        left_hand_velocity = (
            left_hand_pos - previous_spec_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
        )
        right_hand_velocity = (
            right_hand_pos - previous_spec_frame[JOINTS_NAMES_TO_IDX["RightHand"]]
        )

        # Compute energy using the same method as `compute_energy_of_ref_file`
        left_hand_acceleration = left_hand_velocity - prev_left_hand_velocity
        right_hand_acceleration = right_hand_velocity - prev_right_hand_velocity

        left_hand_energy = np.mean(np.abs(left_hand_acceleration)) ** 2
        right_hand_energy = np.mean(np.abs(right_hand_acceleration)) ** 2

        # Store hand energy values for smoothing
        left_hand_energy_history.append(left_hand_energy)
        right_hand_energy_history.append(right_hand_energy)

        # Update previous velocities and frame
        prev_left_hand_velocity = left_hand_velocity
        prev_right_hand_velocity = right_hand_velocity
        previous_spec_frame = spectator_frame

        # Reference angles: Compute the average over a window of reference data
        ref_angles = REF_ANGLES[CURRENT_LEVEL][ref_frame_idx]

        # Send smoothed results at specified intervals
        current_time = time.time()
        if current_time - last_send_time >= RESULT_INTERVAL:
            if (
                spectator_angles_history
                and left_hand_energy_history
                and right_hand_energy_history
            ):
                # Compute smoothed angles and energy
                smoothed_spectator_angles = np.mean(spectator_angles_history, axis=0)
                smoothed_left_hand_energy = np.mean(left_hand_energy_history)
                smoothed_right_hand_energy = np.mean(right_hand_energy_history)

                # Validate choreography using smoothed angles
                choreography_valid = majority_voting(
                    are_angles_close(
                        smoothed_spectator_angles, ref_angles, ANGLES_TOLERANCE
                    )
                )

                # Send results back and log
                client.send_message("/results", bool(choreography_valid))
                client.send_message(
                    "/results",
                    {
                        "choreography_valid": bool(choreography_valid),
                        "left_hand_energy": smoothed_left_hand_energy,
                        "right_hand_energy": smoothed_right_hand_energy,
                    },
                )
                print("Left shoulder", "Left elbow", "Right shoulder", "Right elbow")
                print("Smoothed Spectator Angles:", smoothed_spectator_angles)
                print("Reference Angles:", ref_angles)
                print("Smoothed Left Hand Energy:", smoothed_left_hand_energy)
                print("Smoothed Right Hand Energy:", smoothed_right_hand_energy)
                print("Choreography Valid:", choreography_valid)

                # Clear histories after sending
                spectator_angles_history.clear()
                left_hand_energy_history.clear()
                right_hand_energy_history.clear()

            last_send_time = current_time

    except Exception as e:
        print(f"Error processing data: {e}")


async def loop():
    while True:
        await asyncio.sleep(1)


async def main():
    dispatcher = Dispatcher()
    dispatcher.map("/data*", process_and_send_data)
    dispatcher.map("/level", load_level)
    dispatcher.map("/ping", answer_ping)

    server = AsyncIOOSCUDPServer(
        (OSC_IP, OSC_PORT), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()

    print(f"Server is running on {OSC_IP}:{OSC_PORT}")
    await loop()
    transport.close()


if __name__ == "__main__":
    REF_MOTION = {
        "choreography": extract_ref_motion_data(
            "./choreography_fixed.bvh"
        ),  # from root folder
        "mutation": extract_ref_motion_data("./mutation_dance_fixed.bvh"),
    }
    REF_MOTION = normalize_skeleton(REF_MOTION)
    # Normalize reference skeletons
    REF_ENERGY = compute_energy_of_ref_file(REF_MOTION)
    REF_ANGLES = {}
    for key, val in REF_MOTION.items():
        REF_ANGLES[key] = compute_angles(val, ANGLES_USED_FOR_SCORE)
    asyncio.run(main())


# Ideas to improve: rotate skeleton dynamically so that it's always facing in the right direction to compare
