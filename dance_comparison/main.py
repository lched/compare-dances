import asyncio

try:
    import ujson as json
except ModuleNotFoundError:
    print(
        "Warning: ujson library not found, using native json which is much slower and could cause issues."
    )
    import json
# import threading
from collections import deque

# import cv2
import numpy as np
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

# import cv_viewer.tracking_viewer as cv_viewer
from score import (
    # compute_energy_of_ref_file,
    compute_angles,
    are_angles_close,
    majority_voting,
    compute_hands_energy,
)
from utils import extract_ref_motion_data, convert_angles_names_to_idx
from skeleton_utils import normalize_skeleton, JOINTS_NAMES_TO_IDX


AXES = [0, 2]

# DIFFICULTY SETTINGS
ANGLES_TOLERANCE = 20  # in degrees

# OSC
OSC_PORT = 8080  # PYTHON SERVER PORT
OSC_CLIENT_PORT = 9001  # UNREAL PORT
OSC_IP = "127.0.0.1"
RESULT_INTERVAL = 1  # Interval at which to send results of the analysis (in seconds)

# MISC
USE_3D = False  # TODO

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
    global previous_spec_frame, prev_left_hand_velocity, prev_right_hand_velocitycl
    global spectator_angles_history, left_hand_energy_history, right_hand_energy_history, last_send_time

    try:
        client.send_message("/results", True)
        # timestamp = args[-1]  # in ms
        # ref_frame_idx = int(timestamp / REF_FRAMETIME) % len(REF_MOTION)
        # raw_spectator_frame = np.array(args[:-4]).reshape(-1, 3)
        # # Normalize skeleton
        # spectator_frame = normalize_skeleton(raw_spectator_frame)[:, AXES]  # Work in 2D

        # # Compute angles
        # spectator_angles = compute_angles(
        #     spectator_frame[np.newaxis, :], ANGLES_USED_FOR_SCORE
        # )[0]

        # # Store angles for smoothing
        # spectator_angles_history.append(spectator_angles)

        # # Compute velocities for the hands
        # left_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
        # right_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["RightHand"]]

        # left_hand_velocity = (
        #     left_hand_pos - previous_spec_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
        # )
        # right_hand_velocity = (
        #     right_hand_pos - previous_spec_frame[JOINTS_NAMES_TO_IDX["RightHand"]]
        # )

        # # Compute hand energy
        # left_hand_energy = np.linalg.norm(left_hand_velocity - prev_left_hand_velocity)
        # right_hand_energy = np.linalg.norm(
        #     right_hand_velocity - prev_right_hand_velocity
        # )

        # # Update previous velocities and frames
        # prev_left_hand_velocity = left_hand_velocity
        # prev_right_hand_velocity = right_hand_velocity

        # # Store hand energy values for smoothing
        # left_hand_energy_history.append(left_hand_energy)
        # right_hand_energy_history.append(right_hand_energy)

        # # Reference angles (average over 1 sec of reference data)
        # ref_angles = np.mean(
        #     REF_ANGLES[CURRENT_LEVEL][min(0, ref_frame_idx - 30) : ref_frame_idx]
        # )

        # current_time = time.time()
        # if current_time - last_send_time >= RESULT_INTERVAL:
        #     # Compute the smoothed angles (average over the history)
        #     if spectator_angles_history:
        #         print("history was...", len(spectator_angles_history))
        #         smoothed_spectator_angles = np.mean(spectator_angles_history, axis=0)

        #         # Get the reference angles for the same time period
        #         ref_angles = np.mean(
        #             REF_ANGLES[CURRENT_LEVEL][ref_frame_idx - 30 : ref_frame_idx]
        #         )  # average over 1 sec of reference data

        #         # Check if angles are close enough using smoothed spectator angles
        #         choreography_valid = majority_voting(
        #             are_angles_close(
        #                 smoothed_spectator_angles, ref_angles, ANGLES_TOLERANCE
        #             )
        #         )

        #         # Send results back and log
        #         client.send_message(
        #             "/results",
        #             {
        #                 "choreography_valid": bool(choreography_valid),
        #                 "left_hand_energy": left_hand_energy,
        #                 "right_hand_energy": right_hand_energy,
        #             },
        #         )
        #         print("Smoothed Spectator Angles:", smoothed_spectator_angles)
        #         print("Reference Angles (averaged):", ref_angles)
        #         print("Left Hand Energy:", left_hand_energy)
        #         print("Right Hand Energy:", right_hand_energy)
        #         print("Choreography Valid:", choreography_valid)

        #         # Reset the spectator angles history
        #         spectator_angles_history.clear()
        #         left_hand_energy_history.clear()
        #         right_hand_energy_history.clear()

        #     last_send_time = current_time

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
    # REF_ENERGY = compute_energy_of_ref_file(REF_MOTION[:, used_indices])
    REF_ANGLES = {}
    for key, val in REF_MOTION.items():
        REF_ANGLES[key] = compute_angles(val, ANGLES_USED_FOR_SCORE)
    asyncio.run(main())
