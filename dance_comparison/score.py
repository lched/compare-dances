# Everything related to computing the "score" that says how close we are to the ref choreograpy
import math
import numpy as np

from skeleton_utils import normalize_skeleton


def angle_between_points(A, B, C):
    # Vectors BA and BC
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # Calculate the angle using atan2
    angle_radians = math.atan2(BC[1], BC[0]) - math.atan2(BA[1], BA[0])

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)

    # Normalize the angle to the range [0, 360)
    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees


def are_angles_close(current_angles, ref_angles_history, tolerance):
    """
    Compare current angles to all angles in ref_angles_history (last second of reference data).
    Returns True if any angle in the current frame is within tolerance for any reference frame.
    """
    for ref_angles in ref_angles_history:
        if np.all(np.abs(current_angles - ref_angles) <= tolerance):
            return True
    return False


def compute_energy_of_ref_file(motion_frames):
    """
    Given a temporal dictionary of motion frames, compute the "energy" for each
    sequence in the dictionary. The energy is the absolute value of the
    acceleration for each frame.

    Parameters:
        motion_frames (dict): A dictionary where keys are sequence names
                              (e.g., "choreography") and values are numpy arrays
                              of shape (num_frames, num_joints, 3).

    Returns:
        dict: A dictionary where keys are sequence names and values are
              1D numpy arrays representing the energy for each frame.
    """
    ref_energy = {}

    for key, values in motion_frames.items():
        num_frames = values.shape[0]
        energy = np.zeros(num_frames)

        for i in range(1, num_frames - 1):
            prev_frame = values[i - 1]
            curr_frame = values[i]
            next_frame = values[i + 1]

            # Compute the acceleration as the difference in velocity
            velocity1 = curr_frame - prev_frame
            velocity2 = next_frame - curr_frame
            acceleration = velocity2 - velocity1

            # Sum the absolute value of the acceleration for all joints and square it
            energy[i] = np.mean(np.abs(acceleration)) ** 2

        # Store the energy array for the current sequence
        ref_energy[key] = energy
    return ref_energy


def compute_angles(motion_frames, angle_indices):
    """
    motion_frames shape ()
    angle_indices list of Tuples of 3 indices that make the angles we want to monitor
    """
    angles = np.zeros((motion_frames.shape[0], len(angle_indices)))
    for i in range(len(motion_frames)):
        for j, (A_idx, B_idx, C_idx) in enumerate(angle_indices):
            A = motion_frames[i][A_idx]
            B = motion_frames[i][B_idx]
            C = motion_frames[i][C_idx]
            angles[i][j] = angle_between_points(A, B, C)
    return angles


def majority_voting(results):
    """
    Apply majority voting over the results list (booleans).
    Returns True if more than half the results are True.
    """
    return sum(results) > len(results) // 2


# def compute_hands_energy(skeleton, prev_frame, prev_velocity):
#     """Compute energy for the left and right hands."""
#     left_hand = skeleton[JOINTS_NAMES_TO_IDX["LeftHand"]]
#     right_hand = skeleton[JOINTS_NAMES_TO_IDX["RightHand"]]

#     # Compute velocity (change in position)
#     left_hand_velocity = left_hand - prev_frame[JOINTS_NAMES_TO_IDX["LeftHand"]]
#     right_hand_velocity = right_hand - prev_frame[JOINTS_NAMES_TO_IDX["RightHand"]]

#     # Compute acceleration (change in velocity)
#     left_hand_acceleration = left_hand_velocity - prev_velocity["left"]
#     right_hand_acceleration = right_hand_velocity - prev_velocity["right"]

#     # Compute energy as the sum of squared velocities and accelerations
#     left_hand_energy = np.sum(left_hand_velocity**2 + left_hand_acceleration**2)
#     right_hand_energy = np.sum(right_hand_velocity**2 + right_hand_acceleration**2)

#     # Update previous frame and velocity
#     prev_velocity["left"] = left_hand_velocity
#     prev_velocity["right"] = right_hand_velocity

#     return left_hand_energy, right_hand_energy
