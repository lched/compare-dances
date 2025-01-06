# Everything related to computing the "score" that says how close we are to the ref choreograpy
import math
import numpy as np


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


def are_angles_close(incoming_angles, ref_angles, tolerance=20):
    """
    Check if angles in two arrays are close within a given tolerance.

    Parameters:
        incoming_angles (array-like): Array of incoming angles in degrees.
        ref_angles (array-like): Array of reference angles in degrees.
        tolerance (float): Maximum allowed difference between angles.

    Returns:
        np.ndarray: Boolean array indicating if angles are close.
    """
    # Convert to numpy arrays for element-wise operations
    incoming_angles = np.array(incoming_angles)
    ref_angles = np.array(ref_angles)

    # Calculate the absolute difference between angles
    angle_diff = np.abs(incoming_angles - ref_angles)

    # Adjust for angles that cross the 360° boundary
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)

    # Check if differences are within the tolerance
    return angle_diff <= tolerance


def compute_energy_of_ref_file(motion_frames):
    """Given a temporal array of frames, compute the "energy", which is
    the absolute value of the acceleration for each frame"""
    energy = np.zeros(motion_frames.shape[0])
    for i in range(1, len(motion_frames) - 1):
        prev_frame = motion_frames[i - 1]
        curr_frame = motion_frames[i]
        next_frame = motion_frames[i + 1]

        # Compute the acceleration as the difference in velocity
        velocity1 = curr_frame - prev_frame
        velocity2 = next_frame - curr_frame
        acceleration = velocity2 - velocity1

        # Sum the absolute value of the acceleration
        energy[i] = np.mean(np.abs(acceleration)) ** 2
    return energy


def compute_angles_frame(frame, angle_indices):
    """For a single frame"""
    angles = np.zeros(len(angle_indices))
    for i, (A_idx, B_idx, C_idx) in enumerate(angle_indices):
        A = frame[A_idx]
        B = frame[B_idx]
        C = frame[C_idx]
        angles[i] = angle_between_points(A, B, C)
    return angles


def compute_angles_of_ref_file(motion_frames, angle_indices):
    """For lots of frames
    motion_frames shape ()
    angle_indices list of Tuples of 3 indices that make the angles we want to monitor
    """
    angles = np.zeros((motion_frames.shape[0], len(angle_indices)))
    for i in range(len(motion_frames)):
        angles[i] = compute_angles_frame(motion_frames, angle_indices)
    return angles
