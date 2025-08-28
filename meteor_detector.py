#!/usr/bin/env python3 """ Meteor Detector for Night Videos (e.g., Garmin dashcam)

"""Approach

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


import os import cv2 import math import time import argparse import numpy as np from collections import deque, defaultdict

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def auto_gamma(frame, target=0.25): """Simple auto gamma: map median to target luminance (0..1).""" gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) med = np.median(gray) / 255.0 + 1e-6 gamma = max(0.3, min(3.0, math.log(target, max(med, 1e-4)))) lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8") out = cv2.LUT(frame, lut) return out

def stabilize(prev_gray, gray): """Estimate translation+rotation via feature matching; return stabilized gray and warp matrix.""" orb = cv2.ORB_create(800) kp1, des1 = orb.detectAndCompute(prev_gray, None) kp2, des2 = orb.detectAndCompute(gray, None) if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10: return gray, np.eye(3, dtype=np.float32) bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) matches = bf.match(des1, des2) matches = sorted(matches, key=lambda m: m.distance)[:100] pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) H, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0) if H is None: return gray, np.eye(3, dtype=np.float32) h, w = gray.shape stabilized = cv2.warpAffine(gray, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) H3 = np.eye(3, dtype=np.float32) H3[:2, :] = H return stabilized, H3

def hough_lines(mask, min_len=40, max_gap=4): # Probabilistic Hough; input must be an 8-bit edge image or binary lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=12, minLineLength=min_len, maxLineGap=max_gap) if lines is None: return [] return [tuple(l[0]) for l in lines]

def line_props(x1, y1, x2, y2): dx, dy = x2 - x1, y2 - y1 length = math.hypot(dx, dy) angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360 cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 return length, angle, (cx, cy)

def aspect_of_contour(cnt): x, y, w, h = cv2.boundingRect(cnt) if h == 0: return float("inf") return max(w, h) / max(1, min(w, h))

def iou_rect(r1, r2): x1,y1,w1,h1 = r1 x2,y2,w2,h2 = r2 xa, ya = max(x1,x2), max(y1,y2) xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2) inter = max(0, xb-xa) * max(0, yb-ya) union = w1h1 + w2h2 - inter + 1e-6 return inter/union

class MeteorDetector: def init(self, args): self.args = args self.bg = None  # running average background (float32) self.prebuf = deque(maxlen=args.pre_frames) self.post_countdown = 0 self.writers = [] self.event_id = 0 self.tracks = {}  # id -> {bbox, angle, frames} self.next_track_id = 1

def update_background(self, gray):
    a = self.args.bg_alpha
    if self.bg is None:
        self.bg = gray.astype(np.float32)
    else:
        cv2.accumulateWeighted(gray, self.bg, a)

def detect_candidates(self, gray):
    # residual: positive brightening vs background
    bg8 = self.bg.astype(np.uint8)
    diff = cv2.subtract(gray, bg8)
    # suppress noise with bilateral or median filter
    if self.args.median > 0:
        diff = cv2.medianBlur(diff, self.args.median)
    _, th = cv2.threshold(diff, self.args.thresh, 255, cv2.THRESH_BINARY)
    if self.args.open_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.args.open_kernel, self.args.open_kernel))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    if self.args.close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.args.close_kernel, self.args.close_kernel))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

    # Edge + Hough to enforce linearity
    edges = cv2.Canny(th, 40, 120)
    lines = hough_lines(edges, min_len=self.args.min_length, max_gap=self.args.max_gap)

    candidates = []
    if lines:
        for (x1,y1,x2,y2) in lines:
            length, angle, (cx, cy) = line_props(x1,y1,x2,y2)
            if length < self.args.min_length:
                continue
            # find a local contour near the line to get area/aspect
            mask = np.zeros_like(th)
            cv2.line(mask, (x1,y1), (x2,y2), 255, thickness=max(1, self.args.line_thickness))
            masked = cv2.bitwise_and(th, th, mask=mask)
            cnts,_ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < self.args.min_area:
                continue
            aspect = aspect_of_contour(cnt)
            if aspect < self.args.min_aspect:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            bbox = (x,y,w,h)
            candidates.append({
                'line': (x1,y1,x2,y2),
                'length': length,
                'angle': angle,
                'center': (cx,cy),
                'bbox': bbox,
                'area': area
            })
    return th, candidates

def update_tracks(self, candidates):
    updated = set()
    # Try to match candidates to existing tracks by IoU and angle similarity
    for tid, tr in list(self.tracks.items()):
        best = None
        best_score = 0
        for c in candidates:
            iou = iou_rect(tr['bbox'], c['bbox'])
            dtheta = abs(((tr['angle'] - c['angle'] + 180) % 360) - 180)
            if iou > 0.1 and dtheta < 15:
                score = iou + max(0, 1 - dtheta/15)
                if score > best_score:
                    best_score = score
                    best = c
        if best is not None:
            tr['bbox'] = best['bbox']
            tr['angle'] = best['angle']
            tr['length'] = max(tr['length'], best['length'])
            tr['frames'] += 1
            tr['last_seen'] = 0
            updated.add(id(best))
        else:
            tr['last_seen'] += 1
    # remove stale tracks
    for tid in [t for t,tr in self.tracks.items() if tr['last_seen'] > 2]:
        del self.tracks[tid]
    # create new tracks for unmatched candidates
    for c in candidates:
        if id(c) in updated:
            continue
        self.tracks[self.next_track_id] = {
            'bbox': c['bbox'], 'angle': c['angle'], 'length': c['length'],
            'frames': 1, 'last_seen': 0
        }
        self.next_track_id += 1

    # return confirmed meteor tracks
    confirmed = [ (tid,tr) for tid,tr in self.tracks.items() if tr['frames'] >= self.args.confirm_frames and tr['length'] >= self.args.min_length ]
    return confirmed

def open_writer(self, fourcc, fps, size, out_dir, prefix):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{prefix}_{self.event_id:04d}.mp4")
    self.event_id += 1
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    self.writers.append((writer, path))
    return writer, path

def run(self, video_path, out_dir, save_clips=False, stabilize_flag=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    log_path = os.path.join(out_dir, 'detections.csv')
    ensure_dir(out_dir)
    with open(log_path, 'w') as f:
        f.write('event_id,frame_idx,time_s,length_px,angle_deg,x,y,width,height,clip_path\n')

    writer = None
    clip_path = ''
    last_gray = None
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if self.args.downscale != 1.0:
            frame = cv2.resize(frame, (int(w/self.args.downscale), int(h/self.args.downscale)), interpolation=cv2.INTER_AREA)
            if frame_idx == 0:
                w, h = frame.shape[1], frame.shape[0]
                size = (w, h)

        # preprocess
        if self.args.auto_gamma:
            frame_proc = auto_gamma(frame)
        else:
            frame_proc = frame
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        # optional stabilization
        if stabilize_flag and last_gray is not None:
            gray_stab, _ = stabilize(last_gray, gray)
            gray_use = gray_stab
        else:
            gray_use = gray

        # background update (slower update to avoid consuming meteors)
        self.update_background(gray_use)

        # detect
        th, candidates = self.detect_candidates(gray_use)
        confirmed = self.update_tracks(candidates)

        # keep prebuffer
        self.prebuf.append(frame)

        # draw debug overlay
        debug = frame.copy()
        for c in candidates:
            x1,y1,x2,y2 = c['line']
            cv2.line(debug, (x1,y1), (x2,y2), (0,255,0), 2)
            x,y,w1,h1 = c['bbox']
            cv2.rectangle(debug, (x,y), (x+w1,y+h1), (0,200,255), 1)

        # on confirmation, start or extend clip writing
        if confirmed:
            if save_clips and writer is None:
                writer, clip_path = self.open_writer(fourcc, fps, size, out_dir, 'meteor')
                # dump prebuffer
                for fb in list(self.prebuf):
                    writer.write(fb)
                self.post_countdown = self.args.post_frames
            else:
                self.post_countdown = self.args.post_frames

            # annotate and write current frame
            for tid, tr in confirmed:
                x,y,w1,h1 = tr['bbox']
                cv2.rectangle(debug, (x,y), (x+w1,y+h1), (0,0,255), 2)
                cv2.putText(debug, f"METEOR id={tid}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            if save_clips and writer is not None:
                writer.write(debug)

            # log once per confirmation burst
            with open(log_path, 'a') as f:
                for tid, tr in confirmed:
                    t = frame_idx / max(1e-6, fps)
                    x,y,w1,h1 = tr['bbox']
                    f.write(f"{tid},{frame_idx},{t:.3f},{tr['length']:.1f},{tr['angle']:.1f},{x+w1/2:.1f},{y+h1/2:.1f},{w1},{h1},{clip_path}\n")

        else:
            if save_clips and writer is not None:
                if self.post_countdown > 0:
                    writer.write(debug)
                    self.post_countdown -= 1
                else:
                    writer.release()
                    writer = None
                    clip_path = ''

        # optional debug windows
        if self.args.show:
            cv2.imshow('frame', debug)
            cv2.imshow('residual', th)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        last_gray = gray

    cap.release()
    if self.args.show:
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print(f"Done. Detections logged to {log_path}")

def parse_args(): p = argparse.ArgumentParser(description='Meteor detector for night videos') p.add_argument('--video', required=True, help='Path to input video (MP4/AVI)') p.add_argument('--out', default='detections', help='Output directory') p.add_argument('--auto-gamma', action='store_true', dest='auto_gamma', help='Enable auto gamma normalization') p.add_argument('--bg-alpha', type=float, default=0.01, help='Background update rate (0.005-0.05)') p.add_argument('--thresh', type=int, default=25, help='Brightness residual threshold (8-bit)') p.add_argument('--median', type=int, default=3, help='Median blur kernel (odd, 0 to disable)') p.add_argument('--open-kernel', type=int, default=1, help='Morphological open kernel size (0 to disable)') p.add_argument('--close-kernel', type=int, default=3, help='Morphological close kernel size (0 to disable)') p.add_argument('--min-length', type=int, default=40, help='Minimum Hough line length in pixels') p.add_argument('--max-gap', type=int, default=4, help='Max gap for HoughLinesP') p.add_argument('--line-thickness', type=int, default=2, help='Thickness when validating line region') p.add_argument('--min-area', type=int, default=20, help='Minimum contour area (px) near line') p.add_argument('--min-aspect', type=float, default=6.0, help='Min width/height aspect ratio (thin streak)') p.add_argument('--confirm-frames', type=int, default=2, help='Frames a candidate must persist to confirm') p.add_argument('--pre-frames', type=int, default=15, help='Frames to include before detection in clip') p.add_argument('--post-frames', type=int, default=30, help='Frames to include after last detection in clip') p.add_argument('--save-clips', action='store_true', help='Save MP4 clips for detections') p.add_argument('--downscale', type=float, default=1.0, help='Downscale factor (>1 speeds up, reduces res)') p.add_argument('--stabilize', action='store_true', help='Attempt basic inter-frame stabilization') p.add_argument('--show', action='store_true', help='Show debug windows') return p.parse_args()

def main(): args = parse_args() det = MeteorDetector(args) det.run(args.video, args.out, save_clips=args.save_clips, stabilize_flag=args.stabilize)

if name == 'main': main()
