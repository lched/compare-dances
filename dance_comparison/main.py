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
import matplotlib.pyplot as plt
import numpy as np
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

# import cv_viewer.tracking_viewer as cv_viewer
from score import compute_energy_of_ref_file, compute_angles_of_ref_file
from utils import (
    BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS,
    extract_ref_motion_data,
    IDX_TO_MOTION_FILES,
    BODY38_name2idx,
)


# DATA ABOUT CURRENT LEVEL
CURRENT_LEVEL_IDX = None
ENERGY_ARRAY = None
REF_MOTION = None
REF_FRAMETIME = None
REF_ENERGY = None

# History to compute energy
# Position, velocity, and acceleration tracking
left_hand_history = deque(maxlen=5)
right_hand_history = deque(maxlen=5)
prev_left_hand_velocity = np.zeros(2)
prev_right_hand_velocity = np.zeros(2)
last_left_hand_frame = np.zeros(3)
last_right_hand_frame = np.zeros(3)


# OSC
CV_VIEWER = False
OSC_PORT = 8005
OSC_CLIENT_PORT = 9000  # Port to send responses
OSC_IP = "127.0.0.1"

# MISC
USE_3D = False  # TODO

# SCORE COMPUTATION
JOINTS_USED_FOR_ENERGY = ["LEFT_HAND_MIDDLE_4", "RIGHT_HAND_MIDDLE_4"]
ANGLES_USED_FOR_SCORE = (
    [  # should be tuples of 3 keypoints, the one in the middle being the one with the
        (2, 10, 12),  # Left shoulder
        (2, 11, 13),  # Right shoulder
        (10, 12, 14),  # Left Elbow
        (11, 13, 15),  # Right elbow
    ]
)


client = SimpleUDPClient(OSC_IP, OSC_CLIENT_PORT)  # Create an OSC client


def answer_ping(address, *args):
    print("Received ping!")
    client.send_message("/answer", json.dumps("pong"))


def load_level(address, *args):
    """args[0] is an int that indicates the level"""
    global REF_MOTION, REF_FRAMETIME, REF_ENERGY

    # if CURRENT_LEVEL_IDX == args[0]:
    #     print("Level already loaded")
    #     return
    REF_MOTION, REF_FRAMETIME = extract_ref_motion_data(IDX_TO_MOTION_FILES[args[0]])

    # Compute energy of the keyframes selected in JOINTS_USED_FOR_ENERGY
    used_indices = [
        BODY38_name2idx[keypoint_name] for keypoint_name in JOINTS_USED_FOR_ENERGY
    ]
    REF_ENERGY = compute_energy_of_ref_file(REF_MOTION[:, used_indices])
    REF_ANGLES = compute_angles_of_ref_file(REF_MOTION, ANGLES_USED_FOR_SCORE)

    print(f"Loaded level: {IDX_TO_MOTION_FILES[args[0]]}")


def process_and_send_data(address, *args):
    """Receive data via OSC, compute metrics, and send results back."""

    start_time = time.time()
    try:
        message = json.loads(args[0])
        # Map incoming data to BODY38 format
        mapped_frame = np.zeros((38, 3))
        for body38_idx in range(38):
            fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
            mapped_frame[body38_idx] = message[f"bone{fbx_idx}"]

        # Filter out duplicate frames
        if (mapped_frame[34] == last_left_hand_frame).all() and (
            mapped_frame[35] == last_right_hand_frame
        ).all():
            return

        # Update last seen frames
        last_left_hand_frame = mapped_frame[34]
        last_right_hand_frame = mapped_frame[35]

        # Extract positions of hands
        current_time = time.time() - start_time
        left_hand_position = mapped_frame[34, [1, 2]]  # LEFT_HAND_MIDDLE_4
        right_hand_position = mapped_frame[35, [1, 2]]  # RIGHT_HAND_MIDDLE_4

        # Track position history
        left_hand_history.append((left_hand_position, current_time))
        right_hand_history.append((right_hand_position, current_time))

        # Compute velocity and acceleration
        left_hand_velocity = np.zeros(2)
        left_hand_acceleration = np.zeros(2)
        right_hand_velocity = np.zeros(2)
        right_hand_acceleration = np.zeros(2)

        if len(left_hand_history) > 1:
            pos1, t1 = left_hand_history[-2]
            pos2, t2 = left_hand_history[-1]
            dt = t2 - t1
            if dt > 0:
                left_hand_velocity = (pos2 - pos1) / dt
                left_hand_acceleration = (
                    left_hand_velocity - prev_left_hand_velocity
                ) / dt
                prev_left_hand_velocity = left_hand_velocity

        if len(right_hand_history) > 1:
            pos1, t1 = right_hand_history[-2]
            pos2, t2 = right_hand_history[-1]
            dt = t2 - t1
            if dt > 0:
                right_hand_velocity = (pos2 - pos1) / dt
                right_hand_acceleration = (
                    right_hand_velocity - prev_right_hand_velocity
                ) / dt
                prev_right_hand_velocity = right_hand_velocity

        # # Synchronize with the reference frame
        # reference_frame_index = int(current_time / frame_time) % len(
        #     ref_motion_frames["keypoint_2d"]
        # )
        # reference_frame = ref_motion_frames["keypoint_2d"][reference_frame_index]

        # # Compute similarity to reference frame
        # incoming_angles = calculate_limb_angles(mapped_frame)
        # ref_angles = calculate_limb_angles(reference_frame)
        # # Filter out differences by ignoring indices in IGNORE_LIST
        # filtered_diff = [
        #     abs(ref_angles[j] - incoming_angles[j]) / 180
        #     for j in range(len(LIMB_CONNECTIONS))
        #     if j not in IGNORE_LIST  # Skip indices in IGNORE_LIST
        # ]

        # Prepare the data to send back to the client
        result = {
            "right_hand": {
                "acceleration": right_hand_acceleration.tolist(),
            },
        }
        client.send_message("/result", json.dumps(result))
        print("Sent data")
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
    asyncio.run(main())
