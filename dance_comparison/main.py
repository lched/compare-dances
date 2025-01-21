import asyncio
import time

try:
    import ujson as json
except ModuleNotFoundError:
    print(
        "Warning: ujson library not found, using native json which is much slower and could cause issues."
    )
    import json

import matplotlib.pyplot as plt
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer

from skeleton_utils import normalize_skeleton, JOINTS_NAMES_TO_IDX, PARENTS
from utils import extract_ref_motion_data, get_orthogonal_indices
from draw_utils import draw_skeleton

# Which axes are front-up etc. CHANGES for the incoming data depending on initialization
SPECTATOR_XY_AXES = None

# This should be fixed!
REFERENCE_XY_AXES = [0, 1]

# OSC
OSC_PORT = 8080  # PYTHON SERVER PORT
OSC_CLIENT_PORT = 9001  # UNREAL PORT
OSC_IP = "127.0.0.1"
RESULT_INTERVAL = 1  # Interval at which to send results of the analysis (in seconds)

# DATA ABOUT CURRENT LEVEL
CURRENT_LEVEL = "choreography"
REF_FRAMETIME = 1 / 30  # It should not change!!!!!
REF_MOTION = None
IDX_TO_LEVEL = {0: "choreography", 1: "mutation_dance"}
LEVEL_TO_IDX = {"choreography": 0, "mutation_dance": 1}

# History to compute smoothing
previous_spec_frame = None
last_send_time = time.time()

# Client to send OSC messages back to Unreal
client = SimpleUDPClient(OSC_IP, OSC_CLIENT_PORT)  # Create an OSC client


def answer_ping(address, *args):
    print("Received ping!")
    client.send_message("/answer", json.dumps("pong"))


def load_level(address, *args):
    global CURRENT_LEVEL
    CURRENT_LEVEL = IDX_TO_LEVEL[args[0]]
    print(f"Changed level to {IDX_TO_LEVEL[args[0]]}")


def is_hand_position_close(hand_position, ref_positions, threshold=0.1):
    """Check if the hand position is close to any reference position in 2D."""
    distances = np.linalg.norm(ref_positions - hand_position, axis=1)
    return np.any(distances < threshold)


def mirror_positions(ref_positions):
    """Mirror the reference positions across the vertical axis (x-axis inversion)."""
    mirrored_positions = ref_positions.copy()
    mirrored_positions[:, 0] = -mirrored_positions[:, 0]  # Invert the x-axis
    return mirrored_positions


def process_and_send_data(address, *args):
    """Receive data via OSC, compute metrics, and send results back."""
    global SPECTATOR_XY_AXES, previous_spec_frame, last_send_time, TEMP_INCREMENT

    try:
        # Parse incoming OSC message
        timestamp = args[-1]  # in ms
        ref_frame_idx = int(timestamp / 24)

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
        # ref_frame = REF_MOTION["choreography"][ref_frame_idx][:, REFERENCE_XY_AXES]

        # Get the 2D positions of the spectator's hands
        left_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
        right_hand_pos = spectator_frame[JOINTS_NAMES_TO_IDX["RightHand"]]

        # Get reference positions for the last second
        ref_left_hand_positions = [
            frame[JOINTS_NAMES_TO_IDX["LeftHand"], REFERENCE_XY_AXES]
            for frame in REF_MOTION[CURRENT_LEVEL][
                max(0, ref_frame_idx - 45) : ref_frame_idx
            ]
        ]
        ref_right_hand_positions = [
            frame[JOINTS_NAMES_TO_IDX["RightHand"], REFERENCE_XY_AXES]
            for frame in REF_MOTION[CURRENT_LEVEL][
                max(0, ref_frame_idx - 45) : ref_frame_idx
            ]
        ]

        ref_left_hand_positions = np.vstack(ref_left_hand_positions)
        ref_right_hand_positions = np.vstack(ref_right_hand_positions)

        # Include mirrored positions for comparison
        mirrored_left_hand_positions = mirror_positions(ref_left_hand_positions)
        mirrored_right_hand_positions = mirror_positions(ref_right_hand_positions)

        all_valid_left_positions = np.vstack(
            (ref_left_hand_positions, mirrored_right_hand_positions)
        )
        all_valid_right_positions = np.vstack(
            (ref_right_hand_positions, mirrored_left_hand_positions)
        )

        # Check if hands are close to any reference positions
        left_hand_valid = is_hand_position_close(
            left_hand_pos,
            all_valid_left_positions,
            threshold=THRESHOLDS[CURRENT_LEVEL][ref_frame_idx],
        )
        right_hand_valid = is_hand_position_close(
            right_hand_pos,
            all_valid_right_positions,
            threshold=THRESHOLDS[CURRENT_LEVEL][ref_frame_idx],
        )

        # Send results at specified intervals
        choreography_valid = bool(left_hand_valid and right_hand_valid)
        client.send_message("/results", choreography_valid)
        print(f"{timestamp}/{choreography_valid}")
        # print("Choreography Valid:", choreography_valid)
        # print("Left Hand Valid:", left_hand_valid)
        # print("Right Hand Valid:", right_hand_valid)

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
    REF_MOTION = {key: normalize_skeleton(val) for key, val in REF_MOTION.items()}

    # Different thresholds for different
    choreography_thresholds = np.zeros(REF_MOTION["choreography"].shape[0])
    # First movement (wave)
    choreography_thresholds[0:600] *= 2
    # Second movements
    choreography_thresholds[600:680] *= 2
    # Third movement
    choreography_thresholds[680:985] *= 1.5
    # Fourth movement (static )
    choreography_thresholds[680:1400] *= 1
    # Fifth movement
    choreography_thresholds[1400:1700] *= 3
    # Sixth movement
    choreography_thresholds[1700:1800] *= 1.5
    # Seventh
    choreography_thresholds[1800:2000] *= 1.5
    # Eight
    choreography_thresholds[2000:2190] *= 1.7
    # Ninth
    choreography_thresholds[2190:2278] *= 1.5
    # Tenth (static)
    choreography_thresholds[2278:2450] *= 1
    # 11 (Wave)
    choreography_thresholds[2450:2680] *= 2.5
    # 12 recule et saute
    choreography_thresholds[2680:3030] *= 1.5
    # 13
    choreography_thresholds[3030:] *= 2

    # Mutation should be easy
    mutation_thresholds = np.ones(REF_MOTION["mutation"].shape[0]) * 1
    THRESHOLDS = {
        "choreography": choreography_thresholds,
        "mutation": mutation_thresholds,
    }
    asyncio.run(main())
