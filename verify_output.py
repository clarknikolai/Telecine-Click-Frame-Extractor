#!/usr/bin/env python3
"""
verify_output.py

Scans an output video and flags any frames that may still have
vertical smear artifacts. Reports suspicious frames with their
timecode so you can verify them before sending to the client.

Usage:
    python verify_output.py result.mov
    python verify_output.py result.mov --fps 18
"""

import argparse
import subprocess
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Verify output video for smeared frames")
    ap.add_argument("input", help="Output video file to verify")
    ap.add_argument("--fps", type=int, default=None,
                    help="Override fps for timecode display (auto-detected if not set)")
    args = ap.parse_args()

    # Get fps
    if args.fps:
        fps = args.fps
    else:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=nw=1:nk=1", args.input
        ]).decode().strip()
        num, den = out.split("/")
        fps = int(num) / int(den)

    # Extract all frames as grayscale
    cmd = [
        "ffmpeg", "-v", "error", "-i", args.input,
        "-vf", "scale=320:180",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = 320 * 180
    n_frames = len(raw) // frame_size

    # Compute sharpness for each frame
    scores = []
    for n in range(n_frames):
        start = n * frame_size
        img = np.frombuffer(raw[start:start + frame_size],
                            dtype=np.uint8).reshape(180, 320).astype(np.float32)
        vert_grad = np.abs(np.diff(img, axis=0))
        center = vert_grad[20:160, 40:280]
        scores.append(float(center.mean()))

    scores = np.array(scores)
    median = float(np.median(scores))
    threshold = median * 0.5  # flag anything below 50% of median

    # Find suspicious frames
    suspicious = []
    for i, s in enumerate(scores):
        if s < threshold:
            secs = int(i / fps)
            frame_in_sec = int(i % fps)
            suspicious.append((i, s, f"{secs}s{frame_in_sec:02d}f"))

    # Report
    print(f"File: {args.input}")
    print(f"Total frames: {n_frames}")
    print(f"FPS: {fps}")
    print(f"Sharpness: min={scores.min():.2f} median={median:.2f} max={scores.max():.2f}")
    print(f"Threshold: {threshold:.2f} (50% of median)")
    print()

    if suspicious:
        print(f"WARNING: {len(suspicious)} suspicious frame(s) found:")
        for idx, score, tc in suspicious:
            print(f"  Frame {idx} ({tc}) — sharpness={score:.2f}")
        print()
        print("Check these frames in QuickTime (use arrow keys to step frame-by-frame)")
        print("or in DaVinci Resolve.")
    else:
        print("ALL FRAMES CLEAN — no suspicious frames detected.")

    print()


if __name__ == "__main__":
    main()
