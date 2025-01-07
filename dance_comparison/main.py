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
    compute_energy_of_ref_file,
    compute_angles,
    are_angles_close,
)
from utils import (
    BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS,
    extract_ref_motion_data,
    IDX_TO_MOTION_FILES,
    BODY38_name2idx,
)


# DIFFICULTY SETTINGS
ANGLES_TOLERANCE = 20  # in degrees
SEND_RESULTS_EVERY = 0


# OSC
OSC_PORT = 8080  # PYTHON SERVER PORT
OSC_CLIENT_PORT = 9001  # UNREAL PORT
OSC_IP = "127.0.0.1"
RESULT_INTERVAL = 1  # Interval at which to send results of the analysis (in seconds)

# MISC
USE_3D = False  # TODO

# SCORE COMPUTATION
JOINTS_USED_FOR_ENERGY = ["LEFT_HAND_MIDDLE_4", "RIGHT_HAND_MIDDLE_4"]
ANGLES_USED_FOR_SCORE = (
    [  # should be tuples of 3 keypoints, the one in the middle being the one with the
        (10, 12, 14),  # Left shoulder
        (11, 13, 15),  # Right shoulder
        (12, 14, 16),  # Left elbow
        (13, 15, 17),  # Right elbow
    ]
)


# DATA ABOUT CURRENT LEVEL
CURRENT_LEVEL_IDX = None
ENERGY_ARRAY = None
REF_MOTION = None
REF_FRAMETIME = None
REF_FRAME_IDX = None
REF_ENERGY = None
REF_ANGLES = None

# History to compute energy
# Position, velocity, and acceleration tracking
left_hand_history = deque(maxlen=5)
right_hand_history = deque(maxlen=5)
prev_left_hand_velocity = np.zeros(2)
prev_right_hand_velocity = np.zeros(2)
last_left_hand_frame = np.zeros(3)
last_right_hand_frame = np.zeros(3)
smoothed_results = None
last_send_time = None

# Client to senc OSC messages back to Unreal
client = SimpleUDPClient(OSC_IP, OSC_CLIENT_PORT)  # Create an OSC client


def answer_ping(address, *args):
    print("Received ping!")
    client.send_message("/answer", json.dumps("pong"))


def load_level(address, *args):
    """args[0] is an int that indicates the level"""
    global REF_MOTION, REF_FRAMETIME, REF_ENERGY, REF_ANGLES, CURRENT_LEVEL_IDX, smoothed_results, last_send_time

    if CURRENT_LEVEL_IDX and CURRENT_LEVEL_IDX == args[0]:
        print(f"Level already loaded {IDX_TO_MOTION_FILES[args[0]]}")
        return
    CURRENT_LEVEL_IDX = args[0]
    REF_MOTION, REF_FRAMETIME = extract_ref_motion_data(IDX_TO_MOTION_FILES[args[0]])

    # Initialize buffer to store results depending on the framerate
    smoothed_results = deque(maxlen=int(1 / REF_FRAMETIME * RESULT_INTERVAL))

    # Compute energy of the keyframes selected in JOINTS_USED_FOR_ENERGY
    used_indices = [
        BODY38_name2idx[keypoint_name] for keypoint_name in JOINTS_USED_FOR_ENERGY
    ]
    REF_ENERGY = compute_energy_of_ref_file(REF_MOTION[:, used_indices])
    REF_ANGLES = compute_angles(REF_MOTION, ANGLES_USED_FOR_SCORE)
    last_send_time = time.time()

    print(f"Loaded level: {IDX_TO_MOTION_FILES[args[0]]}")


def process_and_send_data(address, *args):
    """Receive data via OSC, compute metrics, and send results back."""
    global REF_FRAME_IDX, left_hand_history, right_hand_history, prev_left_hand_velocity, prev_right_hand_velocity, last_left_hand_frame, last_right_hand_frame, last_send_time
    start_time = time.time()
    if REF_MOTION is None:
        print("No level loaded! Skipping analysis")
        return
    try:
        timestamp = args[-1]  # in ms
        # print(timestamp)
        if REF_FRAMETIME is not None and REF_MOTION is not None:
            reference_frame_index = int(timestamp / REF_FRAMETIME) % len(REF_MOTION)
        if reference_frame_index != REF_FRAME_IDX:
            REF_FRAME_IDX = reference_frame_index
        data = np.array(args[:-4]).reshape(-1, 3)

        # Map incoming data to BODY38 format
        spectator_frame = np.zeros((38, 3))
        for body38_idx in range(38):
            fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
            spectator_frame[body38_idx] = data[fbx_idx]

        # Update last seen frames
        last_left_hand_frame = spectator_frame[34]
        last_right_hand_frame = spectator_frame[35]

        # Extract positions of hands
        current_time = time.time() - start_time
        left_hand_position = spectator_frame[34, [1, 2]]  # LEFT_HAND_MIDDLE_4
        right_hand_position = spectator_frame[35, [1, 2]]  # RIGHT_HAND_MIDDLE_4

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
        # reference_frame = REF_MOTION[reference_frame_index]

        # # Compute similarity to reference frame
        spectator_energy = (
            np.mean((np.abs(right_hand_acceleration), np.abs(left_hand_acceleration)))
            ** 2
        )
        spectator_angles = compute_angles(
            spectator_frame[np.newaxis, :], ANGLES_USED_FOR_SCORE
        )

        smoothed_results.append(
            {
                "spectator_energy": spectator_energy,
                "spectator_angles": spectator_angles,
                "is_close_angles": are_angles_close(
                    spectator_angles, REF_ANGLES, ANGLES_TOLERANCE
                ).tolist(),
            }
        )

        current_time = time.time()
        if current_time - last_send_time >= RESULT_INTERVAL:
            # Compute smoothed values
            averaged_energy = np.mean(
                [result["spectator_energy"] for result in smoothed_results]
            )
            averaged_angles = np.mean(
                [result["spectator_angles"] for result in smoothed_results], axis=0
            )
            averaged_is_close_angles = np.mean(
                [result["is_close_angles"] for result in smoothed_results], axis=0
            )

            # Prepare the smoothed result
            smoothed_result = {
                "reference_energy": REF_ENERGY[REF_FRAME_IDX],
                "averaged_energy": averaged_energy,
                "reference_angles": REF_ANGLES[REF_FRAME_IDX].tolist(),
                "averaged_angles": averaged_angles.tolist(),
                # "is_close_energy": -1,  # Placeholder if needed
                # "averaged_is_close_angles": (averaged_is_close_angles > 0.5).tolist(),  # Threshold for boolean
            }

            client.send_message("/result", json.dumps(smoothed_result))
            last_send_time = current_time
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
