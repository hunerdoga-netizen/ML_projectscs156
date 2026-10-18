"""Extract the hip-spine angle feature from the tango videos and write a CSV.

For every 5th frame, MediaPipe Pose finds the left shoulder, hip and knee. The angle
at the hip between shoulder and knee is the feature (hip_spine_angle).

Usage:
    python extract_features.py                      # videos in this folder -> tango_posture_dataset.csv
    python extract_features.py --video-dir path/to/videos --output my_dataset.csv
"""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# (file_name, posture_label, clothing_description)
# Labels: 0 = Good Posture, 1 = Bad Posture (Arching), 2 = Bad Posture (Slouching)
VIDEO_JOBS = [
    ("instructorA_good1.MOV", 0, "normal_clothes"),
    ("instructorB_good1.MOV", 0, "regular_sweater"),
    ("instructorB_good1_tightclothes.MOV", 0, "tight_clothes"),
    ("instructorB_good2_tightclothes.MOV", 0, "tight_clothes_2"),
    ("instructorA_bad1.MOV", 1, "normal_clothes"),
    ("instructorB_bad1.MOV", 1, "regular_sweater"),
    ("instructorB_bad1_tightclothes.MOV", 1, "tight_clothes"),
    ("instructorBslouching1_tightclothes.MOV", 2, "tight_clothes_slouching"),
]

FRAME_STEP = 5  # sample every 5th frame


def calculate_angle(a, b, c):
    """Calculates the angle at point b given three points: a, b, c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle


def extract_video(path, video_name, label, clothing, pose, mp_pose, max_frames=None):
    rows = []
    cap = cv2.VideoCapture(path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        frame_count += 1
        if frame_count % FRAME_STEP != 0:
            continue

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue

        lm = results.pose_landmarks.landmark
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        rows.append({
            "video_source": video_name,
            "clothing_type": clothing,
            "shoulder_x": shoulder.x, "shoulder_y": shoulder.y,
            "hip_x": hip.x, "hip_y": hip.y,
            "knee_x": knee.x, "knee_y": knee.y,
            "hip_spine_angle": calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y], [knee.x, knee.y]),
            "posture_label": label,
        })
    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", default=".", help="folder containing the .MOV files")
    parser.add_argument("--output", default="tango_posture_dataset.csv", help="CSV file to write")
    parser.add_argument("--max-frames", type=int, default=None, help="only read the first N frames of each video (for quick tests)")
    args = parser.parse_args()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    all_rows = []
    for video_name, label, clothing in VIDEO_JOBS:
        path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(path):
            print(f"Warning: '{path}' not found. Skipping...")
            continue
        print(f"Processing: {video_name} ({clothing} | Label: {label})...")
        rows = extract_video(path, video_name, label, clothing, pose, mp_pose, args.max_frames)
        all_rows.extend(rows)
        print(f"      Extracted {len(rows)} frames.")

    if not all_rows:
        raise SystemExit("Error: no features were extracted. Check the video folder.")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to '{args.output}'.")
    print(df.groupby("video_source").size())


if __name__ == "__main__":
    main()
