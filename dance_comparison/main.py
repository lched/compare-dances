import json

import cv2
import numpy as np
import time
import pyzed.sl as sl

import cv_viewer.tracking_viewer as cv_viewer

from Motion import BVH
from Motion.Animation import positions_global
from utils import LIMB_CONNECTIONS, SimulatedBodies


USE_ZED = False  # Set to False when you don't have a Zed 2 camera and you just want to see the movements
IGNORE_LIST = (
    []
)  # Joints that are not used for comparison between reference and live movements


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


if __name__ == "__main__":
    # Open ref movements as JSON
    ref_file = "bodies38.json"
    with open(ref_file, "r") as fp:
        ref_motion = json.load(fp)
    ref_frames = list(ref_motion.values())
    ref_timestamps = np.array([f["timestamp"] for f in ref_frames])
    ref_timestamps -= ref_timestamps[0]
    ref_timestamps = ref_timestamps / 1e6  # now contains time in ms?

    bvh_file = "dance_comparison/BP_Mixamo_New10_Scene_1_18_0_fixed.bvh"
    animation, joints_names, frametime = BVH.load(bvh_file)
    anim_xyz = positions_global(animation) * 10 + 500
    bvh_timestamps = np.arange(0, anim_xyz.shape[0], frametime) * 1000
    ref_timestamps = bvh_timestamps

    anim_xyz2d = anim_xyz[:, :, [0, 1]]

    ref_bvh_frames = []
    for frame3d, frame2d in zip(anim_xyz, anim_xyz2d):
        ref_bvh_frames.append(
            {
                "body_list": [
                    {"keypoint": frame3d.tolist(), "keypoint_2d": frame2d.tolist()}
                ]
            }
        )

    if USE_ZED:
        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable Positional tracking (mandatory for object detection)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances
        # positional_tracking_parameters.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_parameters)

        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking = True  # Track people across images flow
        body_param.enable_body_fitting = False  # Smooth skeleton move
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_param.body_format = (
            sl.BODY_FORMAT.BODY_38
        )  # Choose the BODY_FORMAT you wish to use

        # Enable Object Detection module
        zed.enable_body_tracking(body_param)

        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 40

        # Get ZED camera information
        camera_info = zed.get_camera_information()
        # 2D viewer utilities
        display_resolution = sl.Resolution(
            min(camera_info.camera_configuration.resolution.width, 1280),
            min(camera_info.camera_configuration.resolution.height, 720),
        )
        image_scale = [
            display_resolution.width
            / camera_info.camera_configuration.resolution.width,
            display_resolution.height
            / camera_info.camera_configuration.resolution.height,
        ]
    else:
        image_scale = [2 / 3, 2 / 3]
        frame_width, frame_height = [1280, 720]

    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10

    current_ref_frame_idx = 0
    current_ref_timestamp = 0
    n_ref_frames = len(ref_motion.keys())

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
        if USE_ZED and zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            image_render = image.get_data()
        else:
            image_render = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        # Ref body keypoint
        # ref_bodies = SimulatedBodies(ref_frames[current_ref_frame_idx])
        ref_bodies = SimulatedBodies(ref_bvh_frames[current_ref_frame_idx])

        # Update OCV view
        cv_viewer.render_2D(
            image_render,
            image_scale,
            bodies.body_list,
            ref_bodies.body_list,
            sl.BODY_FORMAT.BODY_38,
        )
        if len(ref_bodies.body_list) and len(bodies.body_list) > 0:
            frame_angles = calculate_limb_angles(bodies.body_list[0].keypoint_2d)
            ref_angles = calculate_limb_angles(ref_bodies.body_list[0].keypoint_2d)
            frame_diff = [
                abs(ref_angles[j] - frame_angles[j]) / 180
                for j in range(len(LIMB_CONNECTIONS))
            ]
            score = np.mean(frame_diff)
        cv2.imshow("2D View", image_render)
        # current_ref_frame_idx = (current_ref_frame_idx + 1) % n_ref_frames  # will loop

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
    # viewer.exit()
    image.free(sl.MEM.CPU)
    if USE_ZED:
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()
    cv2.destroyAllWindows()
