from Motion import BVH
from Motion.Animation import positions_global
from skeleton_utils import JOINTS_NAMES_TO_IDX


def extract_ref_motion_data(
    fname,
):
    """Extract motion data from the BVH reference file."""
    animation, _, _ = BVH.load(fname)
    global_position = positions_global(animation)
    # ref_motion = np.zeros((bvh_global_position.shape[0], 38, 3))
    # for body38_idx in range(38):
    #     bvh_idx = BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS[body38_idx]
    #     ref_motion[:, body38_idx] = bvh_global_position[:, bvh_idx]
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
