# MeteorDetect
Approach

1. Stabilize brightness with gamma + denoise.


2. Maintain a running-average background to highlight sudden bright streaks.


3. Threshold positive (brightening) residuals.


4. Validate with edge + probabilistic Hough transform (linear, thin, long).


5. Lightweight multi-frame confirmation (2–4 frames) to reduce false positives.


6. Save event frames and an optional video clip around each detection.



Usage

python meteor_detector.py --video input.mp4 --out out_dir --min-length 40 --min-aspect 6 --confirm-frames 2 --save-clips

Dependencies

opencv-python

numpy


Notes

Works best on night sky footage; for moving cameras, results vary. If your camera moves a lot, consider enabling --stabilize to compensate small ego-motion.

Tune thresholds for your footage (exposure, resolution, FPS). """

Quick start:

Install deps: pip install opencv-python numpy

Run:
python meteor_detector.py --video your_night_clip.mp4 --out detections --auto-gamma --save-clips --confirm-frames 2 --min-length 40

Outputs:

detections/detections.csv with time, angle, length, bbox for each event

Optional MP4 clips around each detection (--save-clips)

Debug windows if you add --show



Tips for dashcam/night sky:

If the camera jiggles, try --stabilize.

If you get false positives from headlights, raise --thresh (e.g., 35–50) and --min-aspect (e.g., 8–12).

For 4K clips, use --downscale 2.0 for speed.
