import json

import cv2
import numpy as np
from enum import Enum
import pyzed.sl as sl
import time

import cv_viewer.tracking_viewer as cv_viewer
from cv_viewer.utils import generate_color_id_u


class SimulatedBodyData:
    def __init__(self, body_info):
        self.id = body_info["id"]
        self.position = body_info["position"]
        self.bounding_box_2d = body_info["bounding_box_2d"]
        self.keypoints = body_info["keypoint"]
        self.keypoint_2d = body_info["keypoint_2d"]
        self.keypoint_confidences = body_info["keypoint_confidence"]
        self.tracking_state = body_info["tracking_state"]


class SimulatedBodies:
    def __init__(self, data):
        self.body_list = [SimulatedBodyData(body) for body in data.get("body_list", [])]


class PoseLandmark(Enum):
    PELVIS = 0
    SPINE_1 = 1
    SPINE_2 = 2
    SPINE_3 = 3
    NECK = 4
    NOSE = 5
    LEFT_EYE = 6
    RIGHT_EYE = 7
    LEFT_EAR = 8
    RIGHT_EAR = 9
    LEFT_CLAVICLE = 10
    RIGHT_CLAVICLE = 11
    LEFT_SHOULDER = 12
    RIGHT_SHOULDER = 13
    LEFT_ELBOW = 14
    RIGHT_ELBOW = 15
    LEFT_WRIST = 16
    RIGHT_WRIST = 17
    LEFT_HIP = 18
    RIGHT_HIP = 19
    LEFT_KNEE = 20
    RIGHT_KNEE = 21
    LEFT_ANKLE = 22
    RIGHT_ANKLE = 23
    LEFT_BIG_TOE = 24
    RIGHT_BIG_TOE = 25
    LEFT_SMALL_TOE = 26
    RIGHT_SMALL_TOE = 27
    LEFT_HEEL = 28
    RIGHT_HEEL = 29
    LEFT_HAND_THUMB_4 = 30
    RIGHT_HAND_THUMB_4 = 31
    LEFT_HAND_INDEX_1 = 32
    RIGHT_HAND_INDEX_1 = 33
    LEFT_HAND_MIDDLE_4 = 34
    RIGHT_HAND_MIDDLE_4 = 35
    LEFT_HAND_PINKY_1 = 36
    RIGHT_HAND_PINKY_1 = 37


LIMB_CONNECTIONS = [
    (PoseLandmark.PELVIS, PoseLandmark.SPINE_1),
    (PoseLandmark.SPINE_1, PoseLandmark.SPINE_2),
    (PoseLandmark.SPINE_2, PoseLandmark.SPINE_3),
    (PoseLandmark.SPINE_3, PoseLandmark.NECK),
    (PoseLandmark.NECK, PoseLandmark.NOSE),
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EAR),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.SPINE_3, PoseLandmark.LEFT_CLAVICLE),
    (PoseLandmark.SPINE_3, PoseLandmark.RIGHT_CLAVICLE),
    (PoseLandmark.LEFT_CLAVICLE, PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_CLAVICLE, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.PELVIS, PoseLandmark.LEFT_HIP),
    (PoseLandmark.PELVIS, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_BIG_TOE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_BIG_TOE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_SMALL_TOE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_SMALL_TOE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HAND_THUMB_4),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_HAND_THUMB_4),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HAND_INDEX_1),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_HAND_INDEX_1),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HAND_MIDDLE_4),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_HAND_MIDDLE_4),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HAND_PINKY_1),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_HAND_PINKY_1),
]


PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    5,
    6,
    7,
    3,
    3,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    0,
    18,
    19,
    20,
    21,
    22,
    23,
    22,
    23,
    22,
    23,
    16,
    17,
    16,
    17,
    16,
    17,
    16,
    17,
]


IGNORE_LIST = [] # Joints that are not used for comparison between reference and live movements


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


# ----------------------------------------------------------------------
#       2D VIEW
# ----------------------------------------------------------------------
def cvt(pt, scale):
    """
    Function that scales point coordinates
    """
    out = [pt[0] * scale[0], pt[1] * scale[1]]
    return out


def render_sk(left_display, img_scale, obj, color, BODY_BONES):
    # Draw skeleton bones
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if (
            kp_a[0] < left_display.shape[1]
            and kp_a[1] < left_display.shape[0]
            and kp_b[0] < left_display.shape[1]
            and kp_b[1] < left_display.shape[0]
            and kp_a[0] > 0
            and kp_a[1] > 0
            and kp_b[0] > 0
            and kp_b[1] > 0
        ):
            cv2.line(
                left_display,
                (int(kp_a[0]), int(kp_a[1])),
                (int(kp_b[0]), int(kp_b[1])),
                color,
                1,
                cv2.LINE_AA,
            )

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]:
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)


def render_2D(left_display, img_scale, objects, body_format):
    """
    Parameters
        left_display (np.array): numpy array containing image data (image shape is for instance (720, 1280, 4))
        img_scale (list[float]) (ex: [0.6666666666666666, 0.6666666666666666])
        objects (list[sl.ObjectData])
    """
    overlay = left_display.copy()

    # Render skeleton joints and bones
    for obj in objects:
        if len(obj.keypoint_2d) > 0:
            color = generate_color_id_u(obj.id)
            if body_format == sl.BODY_FORMAT.BODY_18:
                render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES)
            elif body_format == sl.BODY_FORMAT.BODY_34:
                render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES)
            elif body_format == sl.BODY_FORMAT.BODY_38:
                render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)


if __name__ == "__main__":
    ref_file = "bodies38.json"
    with open(ref_file, "r") as fp:
        ref_motion = json.load(fp)
    ref_frames = list(ref_motion.values())
    ref_timestamps = np.array([f["timestamp"] for f in ref_frames])
    ref_timestamps -= ref_timestamps[0]
    ref_timestamps = ref_timestamps / 1e6  # now contains time in ms?
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
        display_resolution.width / camera_info.camera_configuration.resolution.width,
        display_resolution.height / camera_info.camera_configuration.resolution.height,
    ]

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
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Ref body keypoint
            ref_bodies = SimulatedBodies(ref_frames[current_ref_frame_idx])
            # Update OCV viewqq
            image_render = image.get_data()
            cv_viewer.render_2D(
                image_render,
                image_scale,
                bodies.body_list,
                ref_bodies.body_list,
                body_param.body_format,
            )
            if len(ref_bodies.body_list) and len(bodies.body_list) > 0:
                frame_angles = calculate_limb_angles(bodies.body_list[0].keypoint_2d)
                ref_angles = calculate_limb_angles(ref_bodies.body_list[0].keypoint_2d)
                frame_diff = [
                    abs(ref_angles[j] - frame_angles[j]) / 180
                    for j in range(len(LIMB_CONNECTIONS))
                ]
                score = np.mean(frame_diff)
            cv2.imshow("ZED | 2D View", image_render)
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
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
