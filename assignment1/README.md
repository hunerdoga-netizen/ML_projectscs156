# Tango Posture Classifier (CS156, Pipeline Assignment 1)

Classifies a dancer's posture (good, arching, slouching) from side-view video, using a hip-spine angle computed from pose landmarks.

## Pipeline
1. **Extraction:** MediaPipe Pose finds the left shoulder, hip and knee in every 5th video frame. The angle at the hip between shoulder and knee is the feature (`hip_spine_angle`).
2. **Dataset:** one row per frame, with the video name, clothing type, landmark coordinates, angle and label (0 = good, 1 = arching, 2 = slouching).
3. **Models:** logistic regression, Gaussian naive Bayes and random forest, plus a most-frequent-class baseline.
4. **Comparison conditions:** leave-one-video-out cross-validation, train on Instructor A and test on Instructor B (and the reverse), and a three-class run. Each is run on raw and cleaned data.

## Files
- `assignment-1.ipynb`: the notebook (modeling, figures, write-up).
- `extract_features.py`: turns the videos into the CSV (MediaPipe Pose, hip-spine angle every 5th frame).
- `tango_posture_dataset.csv`: extracted features read by the notebook.
- `tango_factorial_matrices_v4.png`: confusion matrices for all conditions and models.
- `figures/tango_pipeline_diagram.png`: pipeline diagram.

## Running it
```
pip install mediapipe==0.10.14 opencv-python numpy pandas matplotlib seaborn scikit-learn tabulate
```
Open the notebook from inside the `assignment1/` folder so it finds the CSV and the diagram. The raw `.MOV` videos are not included in this repo, so the notebook runs from the CSV.

To regenerate the CSV from the videos (named as listed in `extract_features.py`):
```
python extract_features.py --video-dir path/to/videos --output tango_posture_dataset.csv
```
