# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import time
import numpy as np
from PIL import Image

from qai_hub_models.models.mediapipe_pose.app import MediaPipePoseApp
from qai_hub_models.models.mediapipe_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipePose,
)
from qai_hub_models.utils.args import add_output_dir_arg
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.camera_capture import capture_and_display_processed_frames
from qai_hub_models.utils.display import display_or_save_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pose.jpeg"
)


# Run Mediapipe Pose landmark detection end-to-end on a sample image or camera stream.
# The demo will display output with the predicted landmarks & bounding boxes drawn.
def extract_live_poses(duration):
    # Demo parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="image file path or URL. Image spatial dimensions (x and y) must be multiples",
    )
    add_output_dir_arg(parser)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera Input ID",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.75,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )

    args = parser.parse_args([])
    # if is_test:
    #     args.image = INPUT_IMAGE_ADDRESS

    print(
        "Note: This readme is running through torch, and not meant to be real-time without dedicated ML hardware."
    )
    print("Use Ctrl+C in your terminal to exit.")

    # Load app
    app = MediaPipePoseApp(
        MediaPipePose.from_pretrained(), args.score_threshold, args.iou_threshold
    )
    print("Model and App Loaded")

    if args.image:
        image = load_image(args.image).convert("RGB")
        pred_image = app.predict_landmarks_from_image(image)
        out = Image.fromarray(pred_image[0], "RGB")
        display_or_save_image(out, args.output_dir)
        return np.array(pred_image[0])
    else:
        start_time = time.time()
        captured_poses = []
        def frame_processor(frame: np.ndarray) -> np.ndarray:
            landmarks = app.predict_landmarks_from_image(frame)[0]  # type: ignore
            captured_poses.append(landmarks)
            return landmarks

        while time.time - start_time < duration:
            capture_and_display_processed_frames(
                frame_processor, "QAIHM Mediapipe Pose Demo", args.camera
            )

    return np.array(captured_poses)


# if __name__ == "__main__":
#     main()