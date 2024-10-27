import argparse
import time
import cv2
import numpy as np
from scipy.spatial.distance import cosine
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

# Set a fixed resolution for consistency
TARGET_RESOLUTION = (1280, 480)  

def calculate_accuracy(live_frame, reference_frame):
    """Resize frames to a fixed size and calculate similarity using cosine distance."""
    live_frame_resized = cv2.resize(live_frame, TARGET_RESOLUTION).flatten()
    reference_frame_resized = cv2.resize(reference_frame, TARGET_RESOLUTION).flatten()
    similarity = 1 - cosine(live_frame_resized, reference_frame_resized)
    return similarity * 100  # Convert to percentage

def extract_video_poses(video_path, output_path):
    """Extracts poses from a video and saves processed frames."""
    model = MediaPipePose.from_pretrained()
    app = MediaPipePoseApp(model)

    cap = cv2.VideoCapture(video_path)
    width, height = TARGET_RESOLUTION
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    poses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, TARGET_RESOLUTION)  # Resize to target resolution
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        landmarks = app.predict_landmarks_from_image(rgb_frame)[0].flatten()
        poses.append(landmarks)

        out.write(frame_resized)
        cv2.imshow('Processed Video', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return np.array(poses)

def extract_live_poses(duration, video_poses):
    """Captures live poses for a given duration and compares them with video poses."""
    app = MediaPipePoseApp(MediaPipePose.from_pretrained())

    cap = cv2.VideoCapture(0)  # Use live camera feed
    start_time = time.time()
    frame_count = 0

    while cap.isOpened() and time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, TARGET_RESOLUTION)  # Resize to target resolution
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        landmarks = app.predict_landmarks_from_image(rgb_frame)[0].flatten()

        if frame_count < len(video_poses):
            reference_frame = video_poses[frame_count]
            accuracy = calculate_accuracy(landmarks, reference_frame)

            feedback_text = f"Accuracy: {accuracy:.2f}%"
            color = (0, 255, 0) if accuracy >= 80 else (0, 0, 255)
            cv2.putText(frame_resized, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

        frame_count += 1
        cv2.imshow('Live Feed', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_poses = extract_video_poses('Reference-Video1.mp4', 'output.mp4')
    extract_live_poses(len(video_poses), video_poses)

if __name__ == "__main__":
    main()
