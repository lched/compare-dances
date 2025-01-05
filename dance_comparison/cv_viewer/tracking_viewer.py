import cv2
import numpy as np

from cv_viewer.utils import *
import pyzed.sl as sl


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


def normalize_and_scale_keypoints(keypoints, display_shape, img_scale):
    """
    Normalize and scale keypoints to fit within the display.

    Parameters:
        keypoints (np.array): Original keypoints in 2D space (shape: [n_points, 2])
        display_shape (tuple): Shape of the display (height, width, channels)
        img_scale (list[float]): Scaling factors for the image (ex: [scale_x, scale_y])

    Returns:
        np.array: Transformed keypoints (scaled and resized)
    """
    if len(keypoints) == 0:
        return keypoints

    # Extract display dimensions (height, width)
    display_height, display_width = display_shape[:2]

    # Find bounding box of the skeleton
    min_coords = keypoints.min(axis=0)  # [min_x, min_y]
    max_coords = keypoints.max(axis=0)  # [max_x, max_y]
    bbox_size = max_coords - min_coords

    # Avoid division by zero in case of zero-sized bounding box
    bbox_size[bbox_size == 0] = 1

    # Normalize to [0, 1] based on bounding box
    normalized_keypoints = (keypoints - min_coords) / bbox_size
    return normalized_keypoints
    # Scale to display dimensions (width, height) correctly
    scaled_keypoints = normalized_keypoints * [
        display_width * img_scale[0],
        display_height * img_scale[1],
    ]

    return scaled_keypoints


def render_2D(left_display, img_scale, objects, ref_objects):
    """
    Render 2D skeletons on the display.

    Parameters:
        left_display (np.array): numpy array containing image data (image shape is for instance (720, 1280, 4))
        img_scale (list[float]): Scaling factors for the image
        objects (list[sl.ObjectData]): List of detected objects
        ref_objects (list[sl.ObjectData]): List of reference objects
    """
    overlay = left_display.copy()

    # Render skeleton joints and bones for objects
    for obj in objects:
        if len(obj.keypoint_2d) > 0:
            # Resize keypoints to fit the display
            obj.keypoint_2d = (
                normalize_and_scale_keypoints(
                    obj.keypoint_2d, left_display.shape, img_scale
                )
                * 500
                + 500
            )

            # Render the skeleton
            color = generate_color_id_u(obj.id)
            render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)

    # Render skeleton joints and bones for reference objects
    for obj in ref_objects:
        if len(obj.keypoint_2d) > 0:
            # Resize keypoints to fit the display
            obj.keypoint_2d = (
                normalize_and_scale_keypoints(
                    obj.keypoint_2d, left_display.shape, img_scale
                )
                * 500
                + 500
            )

            # Render the skeleton
            color = generate_color_id_u(obj.id)
            render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)

    # Blend the overlay
    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)
