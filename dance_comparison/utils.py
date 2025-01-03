from enum import Enum


class SimulatedBodyData:
    def __init__(self, body_info):
        # self.id = body_info["id"]
        self.id = 0
        # self.position = body_info["position"]
        # self.bounding_box_2d = body_info["bounding_box_2d"]
        self.keypoints = body_info["keypoint"]
        self.keypoint_2d = body_info["keypoint_2d"]
        # self.keypoint_confidences = body_info["keypoint_confidence"]
        # self.tracking_state = body_info["tracking_state"]


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

BODY38_name2idx = {
    "PELVIS": 0,
    "SPINE_1": 1,
    "SPINE_2": 2,
    "SPINE_3": 3,
    "NECK": 4,
    "NOSE": 5,
    "LEFT_EYE": 6,
    "RIGHT_EYE": 7,
    "LEFT_EAR": 8,
    "RIGHT_EAR": 9,
    "LEFT_CLAVICLE": 10,
    "RIGHT_CLAVICLE": 11,
    "LEFT_SHOULDER": 12,
    "RIGHT_SHOULDER": 13,
    "LEFT_ELBOW": 14,
    "RIGHT_ELBOW": 15,
    "LEFT_WRIST": 16,
    "RIGHT_WRIST": 17,
    "LEFT_HIP": 18,
    "RIGHT_HIP": 19,
    "LEFT_KNEE": 20,
    "RIGHT_KNEE": 21,
    "LEFT_ANKLE": 22,
    "RIGHT_ANKLE": 23,
    "LEFT_BIG_TOE": 24,
    "RIGHT_BIG_TOE": 25,
    "LEFT_SMALL_TOE": 26,
    "RIGHT_SMALL_TOE": 27,
    "LEFT_HEEL": 28,
    "RIGHT_HEEL": 29,
    "LEFT_HAND_THUMB_4": 30,
    "RIGHT_HAND_THUMB_4": 31,
    "LEFT_HAND_INDEX_1": 32,
    "RIGHT_HAND_INDEX_1": 33,
    "LEFT_HAND_MIDDLE_4": 34,
    "RIGHT_HAND_MIDDLE_4": 35,
    "LEFT_HAND_PINKY_1": 36,
    "RIGHT_HAND_PINKY_1": 37,
}

BODY38_idx2name = {
    0: "PELVIS",
    1: "SPINE_1",
    2: "SPINE_2",
    3: "SPINE_3",
    4: "NECK",
    5: "NOSE",
    6: "LEFT_EYE",
    7: "RIGHT_EYE",
    8: "LEFT_EAR",
    9: "RIGHT_EAR",
    10: "LEFT_CLAVICLE",
    11: "RIGHT_CLAVICLE",
    12: "LEFT_SHOULDER",
    13: "RIGHT_SHOULDER",
    14: "LEFT_ELBOW",
    15: "RIGHT_ELBOW",
    16: "LEFT_WRIST",
    17: "RIGHT_WRIST",
    18: "LEFT_HIP",
    19: "RIGHT_HIP",
    20: "LEFT_KNEE",
    21: "RIGHT_KNEE",
    22: "LEFT_ANKLE",
    23: "RIGHT_ANKLE",
    24: "LEFT_BIG_TOE",
    25: "RIGHT_BIG_TOE",
    26: "LEFT_SMALL_TOE",
    27: "RIGHT_SMALL_TOE",
    28: "LEFT_HEEL",
    29: "RIGHT_HEEL",
    30: "LEFT_HAND_THUMB_4",
    31: "RIGHT_HAND_THUMB_4",
    32: "LEFT_HAND_INDEX_1",
    33: "RIGHT_HAND_INDEX_1",
    34: "LEFT_HAND_MIDDLE_4",
    35: "RIGHT_HAND_MIDDLE_4",
    36: "LEFT_HAND_PINKY_1",
    37: "RIGHT_HAND_PINKY_1",
}


BODY38_FORMAT_TO_CORRESPONDING_FBX_KEYPOINTS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 5,
    7: 5,
    8: 5,  # LEFT_EAR
    9: 5,  # RIGHT_EAR
    10: 7,
    11: 31,
    12: 8,
    13: 32,
    14: 9,
    15: 33,
    16: 10,
    17: 34,
    18: 55,
    19: 60,
    20: 56,
    21: 61,
    22: 57,
    23: 62,
    24: 58,  # LEFT_BIG_TOE
    25: 63,  # RIGHT_BIG_TOE
    26: 58,  # LEFT_SMALL_TOE
    27: 63,  # RIGHT_SMALL_TOE
    28: 57,  # LEFT_HEEL
    29: 62,  # RIGHT_HEEL
    30: 13,
    31: 37,
    32: 15,
    33: 39,
    34: 21,
    35: 45,
    36: 27,
    37: 51,
}
