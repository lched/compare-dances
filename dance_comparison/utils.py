from Motion import BVH
from Motion.Animation import positions_global
from skeleton_utils import JOINTS_NAMES_TO_IDX


def extract_ref_motion_data(
    fname, normalize=False,
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
