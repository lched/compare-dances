import numpy as np

from Motion import BVH
from Motion.Animation import positions_global
from skeleton_utils import JOINTS_NAMES_TO_IDX


def extract_ref_motion_data(
    fname,
    normalize=False,
):
    """Extract motion data from the BVH reference file."""
    animation, _, _ = BVH.load(fname)
    global_position = positions_global(animation)
    return global_position


def convert_angles_names_to_idx(angles):
    output = []
    for a in angles:
        output.append(
            (
                JOINTS_NAMES_TO_IDX[a[0]],
                JOINTS_NAMES_TO_IDX[a[1]],
                JOINTS_NAMES_TO_IDX[a[2]],
            )
        )
    return output


def get_orthogonal_indices(left_shoulder, right_shoulder, neck, hips):
    """
    Get the indices of the axes with the largest absolute differences
    between the left and right shoulder.

    Parameters:
        left_shoulder (array-like): XYZ coordinates of the left shoulder.
        right_shoulder (array-like): XYZ coordinates of the right shoulder.

    Returns:
        list: Two indices of the axes with the highest absolute differences.
    """
    # Compute the difference vector between the shoulders
    x_axis = np.argmax(np.abs(np.array(right_shoulder) - np.array(left_shoulder)))
    y_axis = np.argmax(np.abs(np.array(neck) - np.array(hips)))

    return [x_axis, y_axis]
