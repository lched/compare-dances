import asyncio
import websockets

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
IGNORE_LIST = (
    []
)  # Joints that are not used for comparison between reference and live movements
# exclude_list = [8, 9, 24, 25, 26, 27, 28, 29]


def calculate_limb_angles(frame_landmarks):
    """Calculates limb angles for each frame."""

    limb_angles = []
    for start, end in LIMB_CONNECTIONS:
        try:
            start_point = np.array(frame_landmarks[start.value])
            end_point = np.array(frame_landmarks[end.value])

            # calculate angle of limb with respect to vertical (y-axis)
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            # arctan2 identifies sign (i.e. quadrant) + deals with zero values
            angle = np.degrees(np.arctan2(dx, dy))

            # normalize angle -> between 0 and 180 degrees
            angle = abs(angle)
            if angle > 180:
                angle = 360 - angle

            limb_angles.append(angle)
        except (IndexError, ZeroDivisionError):
            # fallback if zero division error
            # shouldn't really happen cus arctan2 deals with this
            limb_angles.append(0)
    return limb_angles


def extract_ref_motion_data(fname):
    animation, joints_names, frametime = BVH.load(fname)
    anim_xyz_fbx = positions_global(animation)
    anim_xyz = np.zeros((anim_xyz_fbx.shape[0], 38, 3))
    for body38_idx in range(38):
        fbx_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
        anim_xyz[:, body38_idx] = anim_xyz_fbx[:, fbx_idx]

    anim_xyz = anim_xyz * 50 + 500
    ref_timestamps = np.arange(0, anim_xyz.shape[0], frametime) * 1000

    anim_xyz2d = anim_xyz[:, :, [0, 1]]

    ref_frames = []
    for frame3d, frame2d in zip(anim_xyz, anim_xyz2d):
        ref_frames.append(
            {
                "body_list": [
                    {"keypoint": frame3d.tolist(), "keypoint_2d": frame2d.tolist()}
                ]
            }
        )

    return ref_timestamps, ref_frames


if __name__ == "__main__":

    async def keypointsVisualizer(websocket):
        try:
            while True:
                # Wait for a message from the server
                message = await websocket.recv()
                print("The python server received a message!")

                if message["type"] == "motion_data":
                    pass
        except websockets.ConnectionClosed:
            print("Client: Server disconnected")

    # Start the server on localhost, port 8765
    start_server = websockets.serve(keypointsVisualizer, "localhost", WEBSOCKET_PORT)
    print(f"Server is running on ws://localhost:{WEBSOCKET_PORT}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

    # Open movements from the reference file
    ref_timestamps, ref_frames = extract_ref_motion_data(REFERENCE_FILE)

    image_scale = [2 / 3, 2 / 3]
    frame_width, frame_height = [1280, 720]

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

        # Grab an image
        image = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        # Received
        received_bodies = SimulatedBodies()
        # Ref body keypoint
        ref_bodies = SimulatedBodies(ref_frames[current_ref_frame_idx])

        # Update viewer
        cv_viewer.render_2D(
            image,
            image_scale,
            received_bodies.body_list,
            ref_bodies.body_list,
        )

        if len(ref_bodies.body_list) and len(received_bodies.body_list) > 0:
            frame_angles = calculate_limb_angles(
                received_bodies.body_list[0].keypoint_2d
            )
            ref_angles = calculate_limb_angles(ref_bodies.body_list[0].keypoint_2d)
            frame_diff = [
                abs(ref_angles[j] - frame_angles[j]) / 180
                for j in range(len(LIMB_CONNECTIONS))
            ]
            score = np.mean(frame_diff)
        cv2.imshow("2D View", np.flipud(image))
        key = cv2.waitKey(key_wait)
        if key == 113:  # for 'q' key
            print("Exiting...")
            break
        if key == 109:  # for 'm' key
            if key_wait > 0:
                print("Pause")
                key_wait = 0
            else:
                print("Restart")
                key_wait = 10
    cv2.destroyAllWindows()
