# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List, Tuple

import torch

from qai_hub_models.models._shared.mediapipe.utils import MediaPipePyTorchAsRoot
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

POSE_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (19, 15),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 16),
    (16, 22),
    (11, 12),
    (12, 24),
    (24, 23),
    (23, 11),
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left ankle and foot
    (28, 30), (30, 32),  # Right ankle and foot
]


# pose detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_DXY, DETECT_DSCALE = (
    -0.2,
    2.0,
)  # Modifiers applied to pose detector output bounding box to encapsulate the entire pose.
POSE_KEYPOINT_INDEX_START = 2  # The pose detector outputs several keypoints. This is the keypoint index for the bottom.
POSE_KEYPOINT_INDEX_END = 3  # The pose detector outputs several keypoints. This is the keypoint index for the top.
ROTATION_VECTOR_OFFSET_RADS = (
    torch.pi / 2
)  # Offset required when computing rotation of the detected pose.

class MediaPipePose(CollectionModel):
    def __init__(self, pose_detector: PoseDetector, pose_landmark_detector: PoseLandmarkDetector) -> None:
        super().__init__()
        self.pose_detector = pose_detector
        self.pose_landmark_detector = pose_landmark_detector

    @classmethod
    def from_pretrained(cls, detector_weights: str = "blazepose.pth", detector_anchors: str = "anchors_pose.npy", landmark_detector_weights: str = "blazepose_landmark.pth") -> MediaPipePose:
        with MediaPipePyTorchAsRoot():
            from blazepose import BlazePose
            from blazepose_landmark import BlazePoseLandmark

            # Ensure full-body model is used
            pose_detector = BlazePose(full_body=True)
            pose_detector.load_weights(detector_weights)
            pose_detector.load_anchors(detector_anchors)

            pose_regressor = BlazePoseLandmark()
            pose_regressor.load_weights(landmark_detector_weights)

            return cls(
                PoseDetector(pose_detector, pose_detector.anchors),
                PoseLandmarkDetector(pose_regressor),
            )



class PoseDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor,
    ):
        super().__init__()
        self.detector = detector
        self.anchors = anchors

    def forward(self, image: torch.Tensor):
        return self.detector(image)

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepose.pth",
        detector_anchors: str = "anchors_pose.npy",
    ):
        with MediaPipePyTorchAsRoot():
            from blazepose import BlazePose

            pose_detector = BlazePose(back_model=True)
            pose_detector.load_weights(detector_weights)
            pose_detector.load_anchors(detector_anchors)
            return cls(pose_detector, pose_detector.anchors)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 128, 128), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["box_coords", "box_scores"]

    @staticmethod
    def get_channel_last_inputs() -> List[str]:
        return ["image"]


class PoseLandmarkDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor):
        output = self.detector(image)
        return output[0], output[1]

    @classmethod
    def from_pretrained(cls, landmark_detector_weights: str = "blazepose_landmark.pth"):
        with MediaPipePyTorchAsRoot():
            from blazepose_landmark import BlazePoseLandmark

            pose_regressor = BlazePoseLandmark()
            pose_regressor.load_weights(landmark_detector_weights)
            cls(pose_regressor)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["scores", "landmarks"]

    @staticmethod
    def get_channel_last_inputs() -> List[str]:
        return ["image"]
    

