import asyncio
import threading
import websockets
import json  # To handle JSON messages
import cv2
import numpy as np
import time

import cv_viewer.tracking_viewer as cv_viewer
from Motion import BVH
from Motion.Animation import positions_global
from utils import (
    LIMB_CONNECTIONS,
    SimulatedBodies,
    BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS,
)

WEBSOCKET_PORT = 8000
REFERENCE_FILE = "dance_comparison/BP_Mixamo_New10_Scene_1_18_0_fixed.bvh"
IGNORE_LIST = []  # Joints not used for comparison [8, 9, 24, 25, 26, 27, 28, 29]

# Shared variables
received_frame = None  # Shared received frame
frame_lock = threading.Lock()  # Lock for synchronizing access to received_frame


def calculate_limb_angles(frame_landmarks):
    """Calculate limb angles for each frame."""
    limb_angles = []
    for start, end in LIMB_CONNECTIONS:
        try:
            start_point = np.array(frame_landmarks[start.value])
            end_point = np.array(frame_landmarks[end.value])

            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = np.degrees(np.arctan2(dx, dy))
            angle = abs(angle) if angle <= 180 else 360 - abs(angle)
            limb_angles.append(angle)
        except (IndexError, ZeroDivisionError):
            limb_angles.append(0)
    return limb_angles


def extract_ref_motion_data(fname):
    """Extract motion data from the BVH reference file."""
    animation, joints_names, frametime = BVH.load(fname)
    anim_xyz_fbx = positions_global(animation)
    anim_xyz = np.zeros((anim_xyz_fbx.shape[0], 38, 3))
    for body38_idx in range(38):
        fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
        anim_xyz[:, body38_idx] = anim_xyz_fbx[:, fbx_idx]
    ref_timestamps = np.arange(0, anim_xyz.shape[0], frametime) * 1000

    anim_xyz2d = anim_xyz[:, :, [0, 1]]

    ref_frames = [
        {
            "body_list": [
                {"keypoint": np.array(frame3d), "keypoint_2d": np.array(frame2d)}
            ]
        }
        for frame3d, frame2d in zip(anim_xyz, anim_xyz2d)
    ]

    return ref_timestamps, ref_frames


def render_frame(ref_timestamps, ref_frames, image_scale, frame_width, frame_height):
    """Real-time rendering of reference and received animation frames."""
    global received_frame  # Use the shared received_frame

    key_wait = 10
    current_ref_frame_idx = 0
    current_ref_timestamp = 0
    start_time = time.time()

    while True:
        elapsed_time = (time.time() - start_time) * 1000
        try:
            if elapsed_time > current_ref_timestamp:
                current_ref_frame_idx += 1
                current_ref_timestamp = ref_timestamps[current_ref_frame_idx]
        except IndexError:
            current_ref_frame_idx = 0
            current_ref_timestamp = ref_timestamps[0]

        # Create an empty image
        image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        # Get the received frame (synchronized)
        received_bodies = None
        with frame_lock:
            if received_frame is not None:
                received_frame = np.array(received_frame)
                received_bodies = SimulatedBodies(
                    {
                        "body_list": [
                            {
                                "keypoint": received_frame,
                                "keypoint_2d": received_frame[:, [0, 1]],
                            }
                        ]
                    }
                )

        # Reference body keypoint
        try:
            ref_bodies = SimulatedBodies(ref_frames[current_ref_frame_idx])
        except IndexError:
            print("Looping animation...")
            current_ref_frame_idx = 0
            ref_bodies = SimulatedBodies(ref_frames[current_ref_frame_idx])

        # Update viewer
        cv_viewer.render_2D(
            image,
            image_scale,
            received_bodies.body_list if received_bodies is not None else [],
            ref_bodies.body_list,
        )

        # Calculate score and display it
        if (
            len(ref_bodies.body_list) > 0
            and received_bodies
            and len(received_bodies.body_list) > 0
        ):
            frame_angles = calculate_limb_angles(
                received_bodies.body_list[0].keypoint_2d
            )
            ref_angles = calculate_limb_angles(ref_bodies.body_list[0].keypoint_2d)
            frame_diff = [
                abs(ref_angles[j] - frame_angles[j]) / 180
                for j in range(len(LIMB_CONNECTIONS))
            ]
            score = np.mean(frame_diff)
            color = (0, 255, 0, 255)  # Color Green
        else:
            score = -1
            color = (0, 0, 255, 255)  # Color Red

        # Display score on the viewer
        image = np.ascontiguousarray(np.flipud(image))
        cv2.putText(
            image,
            f"Score: {score:.2f}",
            (50, 50),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            color,  # Color (Green in RGBA)
            1,  # Thicknessq
            cv2.LINE_AA,  # Line type
        )

        # Display the image
        cv2.imshow("2D View", image)
        key = cv2.waitKey(key_wait)
        if key == 113:  # 'q' key
            print("Exiting...")
            break
        if key == 109:  # 'm' key
            key_wait = 0 if key_wait > 0 else 10

    cv2.destroyAllWindows()


async def keypoints_visualizer(websocket):
    """Receive and visualize motion data from WebSocket."""
    global received_frame  # Use the shared received_frame
    try:
        async for message in websocket:
            print("Message received!")
            try:
                message = json.loads(message)
            except:
                # Decode the JSON message
                message = "{" + message[:-1] + "}"
                message = json.loads(message)
            if "type" in message.keys() and message["type"] == "ping":
                print("that was a ping!")
            else:
                print("Message decoded!")
                mapped_frame = np.zeros((38, 3))
                for body38_idx in range(38):
                    fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[
                        body38_idx
                    ]  # if needs mapping
                    # fbx_idx = body38_idx  # if already BODY38 format
                    mapped_frame[body38_idx] = message[f"bone{fbx_idx}"]
                with frame_lock:  # Synchronize access to received_frame
                    received_frame = mapped_frame.copy()
    except websockets.ConnectionClosed:
        print("WebSocket connection closed.")


def start_websocket_server():
    """Start the WebSocket server in a separate thread."""
    loop = asyncio.new_event_loop()  # Create a new event loop for this thread
    asyncio.set_event_loop(loop)  # Set the event loop for the current thread
    server = websockets.serve(keypoints_visualizer, "localhost", WEBSOCKET_PORT)
    loop.run_until_complete(server)
    print(f"WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
    loop.run_forever()


if __name__ == "__main__":
    # Load reference motion data
    ref_timestamps, ref_frames = extract_ref_motion_data(REFERENCE_FILE)

    # Frame properties
    image_scale = [2 / 3, 2 / 3]
    # image_scale = [1, 1]
    frame_width, frame_height = [1280, 720]

    # Start WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # Start rendering frames
    render_frame(ref_timestamps, ref_frames, image_scale, frame_width, frame_height)
