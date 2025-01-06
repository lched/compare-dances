# Everything related to computing the "score" that says how close we are to the ref choreograpy
import math
import numpy as np


def angle_between_points(A, B, C):
    # Vectors BA and BC
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # Dot product and magnitudes
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    magnitude_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

    # Avoid division by zero
    if magnitude_BA == 0 or magnitude_BC == 0:
        raise ValueError("One of the vectors has zero length.")

    # Cosine of the angle
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)

    # Ensure the value is in the valid range for acos (handling floating-point errors)
    cos_angle = max(-1, min(1, cos_angle))

    # Angle in radians
    angle_radians = math.acos(cos_angle)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def compute_energy(frames):
    """Given a temporal array of frames, compute the "energy", which is
    the absolute value of the acceleration for each frame"""
    energy = np.zeros(frames.shape[0])
    for i in range(1, len(frames) - 1):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        next_frame = frames[i + 1]

        # Compute the acceleration as the difference in velocity
        velocity1 = curr_frame - prev_frame
        velocity2 = next_frame - curr_frame
        acceleration = velocity2 - velocity1

        # Sum the absolute value of the acceleration
        energy[i] = abs(acceleration)
    return energy
