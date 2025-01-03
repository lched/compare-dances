from enum import Enum
import cv2
from statistics import mean


class PoseLandmark(Enum):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


LIMB_CONNECTIONS = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
]


def calculate_limb_angles(
    frame_landmarks: List[List[Tuple[float, float]]]
) -> List[List[float]]:
    """Calculates limb angles for each frame."""
    frame_angles = []

    for landmarks in frame_landmarks:
        limb_angles = []
        for start, end in LIMB_CONNECTIONS:
            try:
                start_point = np.array(landmarks[start.value])
                end_point = np.array(landmarks[end.value])

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
        frame_angles.append(limb_angles)

    return frame_angles


def compare_dancers(
    ref_landmarks: List[List[Tuple[float, float]]],
    comp_landmarks: List[List[Tuple[float, float]]],
    ref_frames: List[np.ndarray],
    comp_frames: List[np.ndarray],
    ref_pose_results: List,
    comp_pose_results: List,
) -> float:
    """Compares two dancers and returns a synchronization score."""
    # get number of comparable frames
    num_frames = min(len(ref_landmarks), len(comp_landmarks))

    ref_angles = calculate_limb_angles(ref_landmarks)
    comp_angles = calculate_limb_angles(comp_landmarks)

    # keep track of current number of out of sync frames (OFS)
    out_of_sync_frames = 0
    score = 100.0

    print("Analysing dancers...")
    video_writer = cv2.VideoWriter(
        f"{OUTPUT_DIR}/output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (2 * 720, 1280),
    )

    for frame_idx in range(num_frames):
        # difference in angle for each limb
        frame_diffs = [
            abs(ref_angles[frame_idx][j] - comp_angles[frame_idx][j]) / 180
            for j in range(len(LIMB_CONNECTIONS))
        ]
        frame_diff = mean(frame_diffs)

        ref_frame = ref_frames[frame_idx]
        comp_frame = comp_frames[frame_idx]

        # annotation skeleton and score on the frame
        mp_draw.draw_landmarks(
            ref_frame,
            ref_pose_results[frame_idx].pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
        mp_draw.draw_landmarks(
            comp_frame,
            comp_pose_results[frame_idx].pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        display = np.concatenate((ref_frame, comp_frame), axis=1)

        color = (0, 0, 255) if frame_diff > SYNC_THRESHOLD else (255, 0, 0)

        cv2.putText(
            display,
            f"Diff: {frame_diff:.2f}",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

        # determine if synced
        if frame_diff > SYNC_THRESHOLD:
            out_of_sync_frames += 1

        score = ((frame_idx + 1 - out_of_sync_frames) / (frame_idx + 1)) * 100.0
        cv2.putText(
            display,
            f"Score: {score:.2f}%",
            (ref_frame.shape[1] + 40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

        cv2.imshow(str(frame_idx), display)
        video_writer.write(display)
        cv2.waitKey(1)

    video_writer.release()
    return score
