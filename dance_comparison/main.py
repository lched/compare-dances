import asyncio
import json
import threading
from collections import deque

import cv2
import numpy as np
import time
import websockets

import cv_viewer.tracking_viewer as cv_viewer
from utils import (
    BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS,
    extract_ref_motion_data,
    IDX_TO_MOTION_FILES,
    LIMB_CONNECTIONS,
    calculate_limb_angles,
)

USE_3D = False  # TODO
CV_VIEWER = False
WEBSOCKET_PORT = 8000
IGNORE_LIST = [
    8,
    9,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
]  # Keypoints not used for comparison


async def process_and_send_data(websocket):
    """Receive data via WebSocket, compute metrics, and send results back."""

    # INITIALIZATTION
    CLOCK = 0
    CURRENT_LEVEL_IDX = 0
    FRAME_COUNT = 0  # Keep track of received frames

    # Load initial reference choreography
    ref_motion_frames, frame_time = extract_ref_motion_data(
        IDX_TO_MOTION_FILES[CURRENT_LEVEL_IDX]
    )

    # Position, velocity, and acceleration tracking
    left_hand_history = deque(maxlen=5)
    right_hand_history = deque(maxlen=5)
    prev_left_hand_velocity = np.zeros(2)
    prev_right_hand_velocity = np.zeros(2)
    last_left_hand_frame = np.zeros(3)
    last_right_hand_frame = np.zeros(3)

    start_time = time.time()

    try:
        async for message in websocket:
            try:
                message = json.loads(message)
            except:
                message = "{" + message[:-1] + "}"
                message = json.loads(message)

            if "type" in message.keys():
                if message["type"] == "ping":
                    print("Received ping")
                    continue
                elif message["type"] == "level":
                    pass  # TODO open new file
                    level_idx = message["data"]
                    if level_idx != CURRENT_LEVEL_IDX:
                        CURRENT_LEVEL_IDX = level_idx
                        ref_motion_frames, frame_time = extract_ref_motion_data(
                            IDX_TO_MOTION_FILES[CURRENT_LEVEL_IDX]
                        )
                elif message["type"] == "start_level":
                    start_time = time.time()  # Reset clock
                    print("Level started")
                else:
                    print(f"Message type not recognized: {message['type']}")
            FRAME_COUNT += 1

            # Map incoming data to BODY38 format
            mapped_frame = np.zeros((38, 3))
            for body38_idx in range(38):
                fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
                mapped_frame[body38_idx] = message[f"bone{fbx_idx}"]

            # Filter out duplicate frames
            if (mapped_frame[34] == last_left_hand_frame).all() and (
                mapped_frame[35] == last_right_hand_frame
            ).all():
                continue

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

            # Synchronize with the reference frame
            reference_frame_index = int(current_time / frame_time) % len(
                ref_motion_frames["keypoint_2d"]
            )
            reference_frame = ref_motion_frames["keypoint_2d"][reference_frame_index]

            # Compute similarity to reference frame
            incoming_angles = calculate_limb_angles(mapped_frame)
            ref_angles = calculate_limb_angles(reference_frame)
            # Filter out differences by ignoring indices in IGNORE_LIST
            filtered_diff = [
                abs(ref_angles[j] - incoming_angles[j]) / 180
                for j in range(len(LIMB_CONNECTIONS))
                if j not in IGNORE_LIST  # Skip indices in IGNORE_LIST
            ]

            # Prepare the data to send back to the client
            result = {
                "left_hand": {
                    "position": left_hand_position.tolist(),
                    "velocity": left_hand_velocity.tolist(),
                    "acceleration": left_hand_acceleration.tolist(),
                },
                "right_hand": {
                    "position": right_hand_position.tolist(),
                    "velocity": right_hand_velocity.tolist(),
                    "acceleration": right_hand_acceleration.tolist(),
                },
            }
            await websocket.send(json.dumps(result))
            if FRAME_COUNT % 10 == 0:
                print("Sent data")
            # print(f"Sent data: {result}")

    except websockets.ConnectionClosed:
        print("WebSocket connection closed.")


def visualize_results(image_scale, frame_width, frame_height):
    """Real-time rendering of received animation frames with FPS estimation."""
    key_wait = 1
    prev_time = time.time()  # Initialize the previous time
    fps_queue = deque(maxlen=30)  # Moving average of the last 30 FPS values
    smoothed_fps = 0

    while True:
        # Measure the time at the start of the frame
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time

        # Create an empty image
        image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        # Calculate FPS and smooth it using a moving average
        if elapsed_time > 0:  # Avoid division by zero
            fps = 1.0 / elapsed_time
            fps_queue.append(fps)
        smoothed_fps = np.mean(fps_queue)

        # Update viewer with FPS
        cv_viewer.render_2D(
            image,
            image_scale,
            [],  # No bodies to display in this version
            [],  # No reference bodies
        )
        image = np.ascontiguousarray(np.flipud(image))
        cv2.putText(
            image,
            f"FPS: {smoothed_fps:.2f}",
            (20, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            (255, 255, 255, 255),  # White color
            1,  # Thickness
            cv2.LINE_AA,  # Line type
        )

        # Display the image
        cv2.imshow("2D View", image)
        key = cv2.waitKey(key_wait)
        if key == 113:  # 'q' key
            print("Exiting...")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start WebSocket server
    loop = asyncio.get_event_loop()
    server = websockets.serve(process_and_send_data, "localhost", WEBSOCKET_PORT)
    loop.run_until_complete(server)

    if CV_VIEWER:
        # Frame properties
        image_scale = [2 / 3, 2 / 3]
        frame_width, frame_height = [1280, 720]
        # Start visualization thread in a separate thread
        rendering_thread = threading.Thread(
            target=visualize_results,
            args=(image_scale, frame_width, frame_height),
            daemon=True,
        )
        rendering_thread.start()
    print(f"Server is running on ws://localhost:{WEBSOCKET_PORT}")
    loop.run_forever()
