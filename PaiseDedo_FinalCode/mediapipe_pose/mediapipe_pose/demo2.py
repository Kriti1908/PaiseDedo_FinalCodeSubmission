import argparse
import cv2
import numpy as np
from qai_hub_models.models.mediapipe_pose.model import MediaPipePose
from qai_hub_models.models.mediapipe_pose.app import MediaPipePoseApp
# give input video path and output video path as commmandline arguments

def extract_video_poses(video_path, output_path):
    # Load the MediaPipePose model and create the app
    model = MediaPipePose.from_pretrained()
    app = MediaPipePoseApp(model)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        processed_frame = app.predict_landmarks_from_image(rgb_frame)[0]

        # Convert back to BGR for writing
        bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(bgr_frame)

        # Display the processed frame
        cv2.imshow('Processed Video', bgr_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return np.array(processed_frame)
# def main():
#     parser = argparse.ArgumentParser(description="Process video with MediaPipePose")
#     parser.add_argument("input_video", help="Path to the input MP4 file")
#     parser.add_argument("output_video", help="Path to save the output MP4 file")
#     args = parser.parse_args()

#     process_video(args.input_video, args.output_video)
#     print(f"Processed video saved to {args.output_video}")

# if __name__ == "__main__":
#     main()