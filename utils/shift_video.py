import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.cons_utils import load_json
import copy


def save_video_with_fps(frames, output_path, frame_size, original_fps, target_fps=None):
    if target_fps is None:
        target_fps = original_fps
    """Save frames into a video file at the specified target frame rate (1 FPS)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to XVID for better compatibility
    out = cv2.VideoWriter(output_path, fourcc, target_fps, frame_size)

    # Calculate the step to downsample frames from original FPS to target FPS
    frame_step = max(1, int(round(original_fps / target_fps)))  # Safeguard against rounding issues

    # Write only frames sampled at the correct step interval
    for i in range(0, len(frames), frame_step):
        out.write(np.uint8(frames[i]))  # Ensure frames are of type np.uint8

    out.release()


def swap_frames(video_path, timestamps, shifted_timestamps, output_dir, vid):
    """Swap frames between original and shifted timestamps and save both original and swapped videos at 1 FPS."""

    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # Buffer to store frames
    original_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Store the frame in the buffer
        original_frames.append(frame)

    # # Save the original video (before swapping) with 1 FPS
    # original_output_path = os.path.join(output_dir, f"{vid}.mp4")
    # save_video_with_fps(original_frames, original_output_path, frame_size, original_fps=fps)

    # Swap frames as per timestamps
    for i, (timestamp, shifted_timestamp) in enumerate(zip(timestamps, shifted_timestamps)):
        swapped_output_path = os.path.join(output_dir, f"{vid}_{i}.mp4")
        frames = copy.deepcopy(original_frames)
        start1, end1 = [int(timestamp[0] * fps), int(timestamp[1] * fps)]
        start2, end2 = [int(shifted_timestamp[0] * fps), int(shifted_timestamp[1] * fps)]

        # Ensure the ranges have the same length by extending or truncating
        length1 = end1 - start1
        length2 = end2 - start2
        if length1 > length2:
            # Extend second range by repeating the last frame
            extra_frames = [frames[end2 - 1]] * (length1 - length2)
            frames2 = np.concatenate([frames[start2:end2], extra_frames], axis=0)
        elif length2 > length1:
            # Truncate second range to match the first range's length
            frames2 = frames[start2:start2 + length1]
        else:
            frames2 = frames[start2:end2]

        # Swap frames between the two ranges
        temp = np.copy(frames[start1:end1])  # Copy frames from first range
        frames[start1:end1] = frames2  # Place frames from second range to first
        frames[start2:start2 + len(temp)] = temp  # Place copied frames into second range

        # Save the swapped video with 1 FPS
        save_video_with_fps(frames, swapped_output_path, frame_size, original_fps=fps)
    cap.release()  # Release outside the loop


if __name__ == "__main__":
    # Load and process each video from annotations
    dset_name = "activitynet" # or "charades
    video_root_path = "" # set the right path to your video files
    annotations = load_json(f"data/{dset_name}_consistency_test.json")
    output_dir = f'{video_root_path}/{dset_name}/shifted_videos/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Output dir:", output_dir)

    for vid, details in tqdm(annotations.items()):
        video_path = f"{video_root_path}/{dset_name}/{vid}.mp4"  # Assuming the videos are in a "data" directory
        if os.path.exists(video_path):
            original_timestamps = details["timestamps"]
            shifted_timestamps = details["shifted_timestamps"]
            swap_frames(video_path, original_timestamps, shifted_timestamps, output_dir, vid)

        else:
            print(f"Video file {video_path} not found.")
