import numpy as np


IDX_TO_JOINTS_NAMES = {
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Head",
    6: "Head_end_site",
    7: "LeftShoulder",
    8: "LeftArm",
    9: "LeftForeArm",
    10: "LeftHand",
    11: "LeftHandIndex1",
    12: "LeftHandIndex2",
    13: "LeftHandIndex3",
    14: "LeftHandIndex3_end_site",
    15: "RightShoulder",
    16: "RightArm",
    17: "RightForeArm",
    18: "RightHand",
    19: "RightHandIndex1",
    20: "RightHandIndex2",
    21: "RightHandIndex3",
    22: "RightHandIndex3_end_site",
    23: "LeftUpLeg",
    24: "LeftLeg",
    25: "LeftFoot",
    26: "LeftToeBase",
    27: "LeftToeBase_end_site",
    28: "RightUpLeg",
    29: "RightLeg",
    30: "RightFoot",
    31: "RightToeBase",
    32: "RightToeBase_end_site",
}


JOINTS_NAMES_TO_IDX = {
    "Hips": 0,
    "Spine": 1,
    "Spine1": 2,
    "Spine2": 3,
    "Neck": 4,
    "Head": 5,
    "Head_end_site": 6,
    "LeftShoulder": 7,
    "LeftArm": 8,
    "LeftForeArm": 9,
    "LeftHand": 10,
    "LeftHandIndex1": 11,
    "LeftHandIndex2": 12,
    "LeftHandIndex3": 13,
    "LeftHandIndex3_end_site": 14,
    "RightShoulder": 15,
    "RightArm": 16,
    "RightForeArm": 17,
    "RightHand": 18,
    "RightHandIndex1": 19,
    "RightHandIndex2": 20,
    "RightHandIndex3": 21,
    "RightHandIndex3_end_site": 22,
    "LeftUpLeg": 23,
    "LeftLeg": 24,
    "LeftFoot": 25,
    "LeftToeBase": 26,
    "LeftToeBase_end_site": 27,
    "RightUpLeg": 28,
    "RightLeg": 29,
    "RightFoot": 30,
    "RightToeBase": 31,
    "RightToeBase_end_site": 32,
}


PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    3,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    3,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    0,
    23,
    24,
    25,
    26,
    0,
    28,
    29,
    30,
    31,
]


def normalize_skeleton(skeleton):
    """
    Normalize the skeleton for size and position. Supports both single frames
    and sequences of frames, as well as dictionaries of sequences.

    Parameters:
        skeleton (np.ndarray or dict):
            - A single skeleton frame (shape: (n_joints, 2) or (n_joints, 3)).
            - A sequence of skeleton frames (shape: (n_frames, n_joints, 2) or
              (n_frames, n_joints, 3)).
            - A dictionary where keys are sequence names and values are sequences of frames.

    Returns:
        np.ndarray or dict:
            - Normalized skeleton for single frame or sequence.
            - A dictionary with normalized skeleton sequences if input is a dictionary.
    """
    if isinstance(skeleton, dict):
        # If input is a dictionary, normalize each sequence
        return {key: normalize_skeleton(value) for key, value in skeleton.items()}

    elif isinstance(skeleton, np.ndarray):
        if len(skeleton.shape) == 2:
            # Single frame: (n_joints, 2) or (n_joints, 3)
            return _normalize_single_frame(skeleton)

        elif len(skeleton.shape) == 3:
            # Sequence of frames: (n_frames, n_joints, 2) or (n_frames, n_joints, 3)
            return np.array([_normalize_single_frame(frame) for frame in skeleton])

        else:
            raise ValueError(
                "Input array must have shape (n_joints, d) or (n_frames, n_joints, d), where d is 2 or 3."
            )

    else:
        raise TypeError("Input must be a numpy array or a dictionary of numpy arrays.")


def _normalize_single_frame(skeleton):
    """
    Normalize a single skeleton frame.

    Parameters:
        skeleton (np.ndarray): Skeleton frame (shape: (n_joints, 2) or (n_joints, 3)).

    Returns:
        np.ndarray: Normalized skeleton.
    """
    root = skeleton[JOINTS_NAMES_TO_IDX["Hips"]]

    # Translate so that the root joint is at the origin
    skeleton_translated = skeleton - root

    # Compute skeleton height (distance between root and farthest joint)
    joint_distances = np.linalg.norm(skeleton_translated, axis=1)
    skeleton_height = np.max(joint_distances)

    # Scale the skeleton to a standard size (e.g., height of 1)
    skeleton_normalized = skeleton_translated / skeleton_height

    return skeleton_normalized
