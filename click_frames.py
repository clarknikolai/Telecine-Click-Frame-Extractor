#!/usr/bin/env python3
"""
click_frames.py — Telecine Click-Frame Extractor
Created by Khubchand Bansal for Clark Nikolai.

Extracts clean film frames from a telecine video capture by detecting
audio click/pulse events and selecting the corresponding video frames.

Designed for use with MovieStuff telecine units (and similar machines)
that embed a sync pulse in the audio track for each film frame. The
script detects these pulses, maps them to video frames, and outputs
a new video (or still images) containing only the film frames —
removing all inter-frame transition artifacts.

Features:
  - Audio pulse detection (envelope + rising-edge)
  - Smart frame selection with sharpness analysis
  - Deinterlacing (single-field extraction)
  - Configurable output frame rate (16, 18, 24 fps)
  - Frame offset adjustment
  - Vertical flip for upside-down telecine output
  - Section processing (process just a segment for testing)
  - Auto-adaptive smear filtering
  - ProRes 422 output matching source codec
  - Streaming architecture — works with any file size (tested up to 24GB)

Usage examples:

  # Basic: extract click frames at 18fps, deinterlaced (bottom field)
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart

  # Regular 8mm at 16fps
  python click_frames.py --input reg8_capture.mov --output film.mov \\
      --mode video --out-fps 16 --field bottom --no-smart

  # Super 8 at 18fps
  python click_frames.py --input super8_capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart

  # Keep interlaced (for 30p source or when both fields are wanted)
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field both --no-smart

  # With smart sharpness-based frame selection
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom

  # Extract as numbered PNG stills
  python click_frames.py --input capture.mov --output frames/ \\
      --mode images

  # Adjust frame offset (+1 frame later)
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart --offset 1

  # Process only first 30 seconds (for quick testing)
  python click_frames.py --input capture.mov --output test.mov \\
      --mode video --out-fps 18 --field bottom --no-smart --duration 30

  # Process a 30-second segment starting at 60 seconds
  python click_frames.py --input capture.mov --output test.mov \\
      --mode video --out-fps 18 --field bottom --no-smart \\
      --start 60 --duration 30

  # Flip upside-down telecine output
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart --flip-vertical

Dependencies:
  pip install numpy
  ffmpeg and ffprobe must be installed and on PATH

Tested with:
  - MovieStuff HD telecine (8mm/Super 8)
  - Canon HV30 capture at 59.94i
  - ProRes 422 LT, 1920x1080, 29.97fps source files

Author: Khubchand
License: MIT
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction

import numpy as np


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def ffprobe_video_info(path):
    """Probe video file and return fps, frame count, resolution, codec."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.check_output(cmd).decode()
    info = json.loads(out)
    stream = info["streams"][0]

    r_frame_rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate")
    fps_frac = Fraction(r_frame_rate)
    fps = float(fps_frac)

    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        nb_frames = int(nb_frames)
    else:
        dur = float(stream.get("duration") or info["format"]["duration"])
        nb_frames = int(round(dur * fps))

    return {
        "fps": fps,
        "fps_frac": fps_frac,
        "nb_frames": nb_frames,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
    }


# ---------------------------------------------------------------------------
# Audio click detection
# ---------------------------------------------------------------------------

def extract_audio_mono(video_path, sr=48000, start=None, duration=None):
    """Extract mono float32 audio from video via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio, sr


def detect_clicks(audio, sr, threshold=0.3, min_gap_s=0.02):
    """
    Detect click/pulse events using envelope rising-edge detection.

    1. Rectify the signal and smooth with a 2ms window.
    2. Threshold at a fraction of peak envelope.
    3. Find rising edges (silence -> pulse transitions).
    4. Enforce minimum gap between consecutive pulses.

    Returns a sorted list of pulse timestamps in seconds.
    """
    if audio.size == 0:
        return []

    env = np.abs(audio)
    win = max(1, int(sr * 0.002))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(env, kernel, mode="same")

    env_max = float(env.max())
    if env_max <= 0:
        return []

    thr = threshold * env_max
    above = env > thr
    rising = np.where(np.diff(above.astype(np.int8)) == 1)[0] + 1

    if rising.size == 0:
        return []

    min_gap = int(min_gap_s * sr)
    kept = [int(rising[0])]
    for p in rising[1:]:
        if int(p) - kept[-1] >= min_gap:
            kept.append(int(p))

    return [p / sr for p in kept]


# ---------------------------------------------------------------------------
# Timestamp -> frame index
# ---------------------------------------------------------------------------

def timestamps_to_frames(timestamps, fps, nb_frames, offset=0):
    """Convert audio timestamps to video frame indices with optional offset."""
    frames = []
    seen = set()
    for t in timestamps:
        idx = int(round(t * fps)) + offset
        if idx < 0 or idx >= nb_frames:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        frames.append(idx)
    frames.sort()
    return frames


# ---------------------------------------------------------------------------
# Sharpness-based frame selection
# ---------------------------------------------------------------------------

def compute_sharpness_scores(video_path, nb_frames,
                              start=None, duration=None):
    """
    Compute vertical sharpness for all frames in the video.

    Higher score = sharper image (clean film frame).
    Lower score = vertical smearing (film was moving during exposure).

    Uses vertical gradients in the center region of a downscaled
    grayscale version of each frame.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", "scale=320:180",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = 320 * 180
    actual = len(raw) // frame_size

    scores = {}
    for n in range(min(actual, nb_frames)):
        start = n * frame_size
        img = np.frombuffer(raw[start:start + frame_size],
                            dtype=np.uint8).reshape(180, 320).astype(np.float32)
        vert_grad = np.abs(np.diff(img, axis=0))
        center = vert_grad[20:160, 40:280]
        scores[n] = float(center.mean())

    return scores


def pick_cleanest_frames(timestamps, fps, nb_frames, video_path,
                         search_range=2, max_score=None,
                         start=None, duration=None):
    """
    For each audio click, look at nearby frames and pick the sharpest one.

    Optionally filters out frames with very low sharpness (smeared or
    leader/black frames) using either an absolute threshold or an
    auto-adaptive threshold based on the file's own distribution.
    """
    print(f"      Analyzing all {nb_frames} frames for sharpness...")
    scores = compute_sharpness_scores(video_path, nb_frames,
                                       start=start, duration=duration)

    # First pass: pick sharpest frame near each click
    pre_picked = []
    seen = set()
    for t in timestamps:
        center = int(round(t * fps))
        cands = [center + off for off in range(-search_range, search_range + 1)
                 if 0 <= center + off < nb_frames]
        if not cands:
            continue
        best = max(cands, key=lambda n: scores.get(n, 0))
        if best not in seen:
            seen.add(best)
            pre_picked.append(best)

    # Second pass: filter outliers
    if max_score is not None and max_score == 0:
        picked = pre_picked
    elif max_score is not None and max_score > 0:
        picked = [f for f in pre_picked if scores.get(f, 0) >= max_score]
        rejected = len(pre_picked) - len(picked)
        if rejected > 0:
            print(f"      Rejected {rejected} frames "
                  f"(sharpness < {max_score})")
    else:
        picked_scores = [scores.get(f, 0) for f in pre_picked]
        if picked_scores:
            median_sharp = float(np.median(picked_scores))
            auto_thr = median_sharp * 0.4
            picked = [f for f in pre_picked
                      if scores.get(f, 0) >= auto_thr]
            rejected = len(pre_picked) - len(picked)
            if rejected > 0:
                print(f"      Auto-filtered {rejected} frame(s) "
                      f"(sharpness < {auto_thr:.1f}, "
                      f"40% of median {median_sharp:.1f})")
        else:
            picked = pre_picked

    picked.sort()
    return picked


# ---------------------------------------------------------------------------
# ffmpeg extraction (streaming approach — works for any file size)
# ---------------------------------------------------------------------------

def extract_images(video_path, frame_indices, out_dir, info=None,
                    field="both", flip_vertical=False,
                    start=None, duration=None):
    """Extract selected frames as numbered PNG stills using streaming."""
    os.makedirs(out_dir, exist_ok=True)
    if not frame_indices:
        return

    keep = set(frame_indices)
    w = info["width"] if info else 1920
    h = info["height"] if info else 1080

    read_cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        read_cmd += ["-ss", str(start)]
    read_cmd += ["-i", video_path]
    if duration is not None:
        read_cmd += ["-t", str(duration)]

    vf_parts = []
    if field in ("top", "bottom"):
        field_val = "0" if field == "top" else "1"
        vf_parts.append(f"field={field_val}")
        out_h = h // 2
    else:
        out_h = h
    if flip_vertical:
        vf_parts.append("vflip")
    if vf_parts:
        read_cmd += ["-vf", ",".join(vf_parts)]
    read_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    frame_size = w * out_h * 3
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)

    count = 0
    n = 0
    while True:
        raw = reader.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if n in keep:
            count += 1
            out_path = os.path.join(out_dir, f"click_{count:05d}.png")
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{out_h}", "-i", "-", out_path],
                stdin=subprocess.PIPE,
            )
            enc.stdin.write(raw)
            enc.stdin.close()
            enc.wait()
        n += 1

    reader.stdout.close()
    reader.wait()


def extract_video(video_path, frame_indices, out_path, info,
                   out_fps=None, field="both", flip_vertical=False,
                   start=None, duration=None):
    """
    Build output video from selected frames using streaming.

    Reads the source frame-by-frame (with optional ffmpeg deinterlacing
    and vertical flip), keeps only the selected frames, and pipes them
    to an ffmpeg encoder. This avoids the select filter expression
    length limit and works for any file size including multi-GB files.
    """
    if not frame_indices:
        return

    keep = set(frame_indices)
    fps_val = out_fps if out_fps else info["fps"]
    fps_str = str(int(fps_val)) if fps_val == int(fps_val) else str(fps_val)

    w = info["width"]
    h = info["height"]

    # Build reader command — let ffmpeg handle deinterlacing and flip
    read_cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        read_cmd += ["-ss", str(start)]
    read_cmd += ["-i", video_path]
    if duration is not None:
        read_cmd += ["-t", str(duration)]

    vf_parts = []
    if field in ("top", "bottom"):
        field_val = "0" if field == "top" else "1"
        vf_parts.append(f"field={field_val}")
        out_h = h // 2
    else:
        out_h = h
    if flip_vertical:
        vf_parts.append("vflip")
    if vf_parts:
        read_cmd += ["-vf", ",".join(vf_parts)]
    read_cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    # Build encoder command
    enc_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{out_h}",
        "-r", fps_str,
        "-i", "-",
        "-an",
    ]
    if field == "both":
        enc_cmd += ["-flags", "+ilme+ildct", "-field_order", "tt"]
    enc_cmd += ["-c:v", "prores_ks", "-profile:v", "2", out_path]

    frame_size = w * out_h * 3  # RGB24

    # Start reader and encoder
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    written = 0
    n = 0
    try:
        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            if n in keep:
                encoder.stdin.write(raw)
                written += 1
            n += 1
    finally:
        encoder.stdin.close()
        reader.stdout.close()
        encoder.wait()
        reader.wait()

    if written == 0:
        print("      WARNING: no frames were written!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Telecine Click-Frame Extractor — extract clean film "
                    "frames from telecine video captures using audio sync pulses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Super 8 at 18fps, deinterlaced (bottom field)
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart

  # Regular 8mm at 16fps
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 16 --field bottom --no-smart

  # Keep interlaced (for 30p captures)
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field both --no-smart

  # With smart sharpness-based frame selection
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom

  # Extract as numbered PNG stills
  python click_frames.py --input capture.mov --output frames/ --mode images

  # Process only first 30 seconds (quick test)
  python click_frames.py --input capture.mov --output test.mov \\
      --mode video --out-fps 18 --field bottom --no-smart --duration 30

  # Vertical flip for upside-down telecine
  python click_frames.py --input capture.mov --output film.mov \\
      --mode video --out-fps 18 --field bottom --no-smart --flip-vertical
        """,
    )
    ap.add_argument("--input", required=True,
                    help="Input video file (telecine capture)")
    ap.add_argument("--output", required=True,
                    help="Output video file or folder (for images mode)")
    ap.add_argument("--mode",
                    choices=["images", "video", "compare"],
                    default="video",
                    help="Output type: 'video' = single video file, "
                         "'images' = numbered PNG stills, "
                         "'compare' = both click and non-click videos "
                         "(default: video)")
    ap.add_argument("--out-fps", type=float, default=None,
                    choices=[16, 18, 24],
                    help="Output frame rate: 16 (Regular 8), 18 (Super 8), "
                         "or 24 (standard film). If not set, uses source fps.")
    ap.add_argument("--field", choices=["top", "bottom", "both"],
                    default="both",
                    help="Field selection for deinterlacing: "
                         "'top' = upper field only (1920x540), "
                         "'bottom' = lower field only (1920x540), "
                         "'both' = keep interlaced (1920x1080). "
                         "(default: both)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Frame offset from click position: "
                         "-1 = one frame earlier, +1 = one frame later "
                         "(default: 0). Only used with --no-smart.")
    ap.add_argument("--no-smart", action="store_true",
                    help="Disable smart frame selection. Use direct "
                         "click-to-frame mapping (recommended when "
                         "using --field top/bottom for deinterlacing).")
    ap.add_argument("--search-range", type=int, default=2,
                    help="Frames to search around each click for the "
                         "sharpest frame (default: 2). Only used with "
                         "smart selection.")
    ap.add_argument("--max-score", type=float, default=None,
                    help="Minimum sharpness threshold. Frames below this "
                         "are rejected. If not set, uses auto-adaptive "
                         "filtering. Set to 0 to disable filtering.")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Audio click detection sensitivity (0-1). "
                         "Lower = more sensitive (default: 0.3).")
    ap.add_argument("--min-gap", type=float, default=0.02,
                    help="Minimum seconds between detected clicks "
                         "(default: 0.02)")
    ap.add_argument("--log", default=None,
                    help="Write detected click timestamps and frame "
                         "indices to this file")
    ap.add_argument("--flip-vertical", action="store_true",
                    help="Flip the output video vertically "
                         "(useful when telecine produces upside-down film)")
    ap.add_argument("--start", type=float, default=None,
                    help="Start time in seconds. Process only from this "
                         "point onwards (default: start of file).")
    ap.add_argument("--duration", type=float, default=None,
                    help="Duration in seconds to process. Useful for "
                         "testing on a short segment (e.g. --duration 30 "
                         "to process only first 30 seconds).")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ERROR: ffmpeg and ffprobe must be installed and on PATH.")

    # Step 1: Probe video
    print(f"[1/5] Probing video: {args.input}")
    info = ffprobe_video_info(args.input)
    print(f"      {info['width']}x{info['height']}  "
          f"fps={info['fps']:.5f} ({info['fps_frac']})  "
          f"frames={info['nb_frames']}  codec={info['codec']}")

    # Adjust frame count for section processing
    section_frames = info["nb_frames"]
    if args.duration is not None:
        section_frames = min(section_frames,
                             int(round(args.duration * info["fps"])))
    if args.start is not None or args.duration is not None:
        start_s = args.start if args.start else 0
        dur_s = args.duration if args.duration else (
            info["nb_frames"] / info["fps"] - start_s)
        print(f"      Section: {start_s:.2f}s - {start_s + dur_s:.2f}s "
              f"(~{section_frames} frames)")

    # Step 2: Detect audio clicks
    print("[2/5] Extracting audio and detecting clicks...")
    audio, sr = extract_audio_mono(args.input,
                                    start=args.start,
                                    duration=args.duration)
    timestamps = detect_clicks(audio, sr,
                               threshold=args.threshold,
                               min_gap_s=args.min_gap)
    print(f"      Detected {len(timestamps)} click(s)")

    # Step 3: Map clicks to frames
    if args.no_smart:
        print(f"[3/5] Using fixed offset={args.offset:+d} "
              f"(smart selection OFF)")
        frames = timestamps_to_frames(timestamps, info["fps"],
                                      section_frames,
                                      offset=args.offset)
    else:
        print(f"[3/5] Smart frame selection (picking cleanest frame "
              f"within +/-{args.search_range} of each click)...")
        frames = pick_cleanest_frames(timestamps, info["fps"],
                                      section_frames, args.input,
                                      search_range=args.search_range,
                                      max_score=args.max_score,
                                      start=args.start,
                                      duration=args.duration)

    click_set = set(frames)
    non_click_frames = [i for i in range(section_frames)
                        if i not in click_set]

    total = section_frames
    n_click = len(frames)
    n_non = len(non_click_frames)
    print("      ------------------------------------------------")
    print(f"      Total frames in source video : {total}")
    print(f"      Click frames (selected)      : {n_click}")
    print(f"      Non-click frames             : {n_non}")
    print(f"      (selected + non = {n_click + n_non})")
    print("      ------------------------------------------------")

    if args.log:
        with open(args.log, "w") as f:
            f.write(f"# total_frames={total} click_frames={n_click} "
                    f"non_click_frames={n_non}\n")
            f.write("# click_index\tframe_index\n")
            for i, fr in enumerate(frames):
                f.write(f"{i}\t{fr}\n")
        print(f"      Log written to {args.log}")

    if not frames:
        print("No click frames detected. Try lowering --threshold.")
        return

    # Step 4: Write output
    out_fps = args.out_fps
    field = args.field
    fps_label = f"{int(out_fps)}fps" if out_fps else "source fps"
    field_label = f", field={field}" if field != "both" else ""
    flip_label = ", vflip" if args.flip_vertical else ""
    print(f"[4/5] Writing output ({args.mode}, "
          f"{fps_label}{field_label}{flip_label})...")

    common = dict(out_fps=out_fps, field=field,
                   flip_vertical=args.flip_vertical,
                   start=args.start, duration=args.duration)

    if args.mode == "images":
        extract_images(args.input, frames, args.output, info=info,
                       field=field, flip_vertical=args.flip_vertical,
                       start=args.start, duration=args.duration)
        print(f"      Wrote {len(frames)} PNG(s) to {args.output}/")
    elif args.mode == "video":
        extract_video(args.input, frames, args.output, info, **common)
        print(f"      Wrote {args.output}")
    else:  # compare
        base, ext = os.path.splitext(args.output)
        if not ext:
            ext = ".mov"
        click_out = f"{base}_clicks{ext}"
        non_out = f"{base}_nonclicks{ext}"
        print(f"      -> click-only video     : {click_out}")
        extract_video(args.input, frames, click_out, info, **common)
        if non_click_frames:
            print(f"      -> non-click video      : {non_out}")
            extract_video(args.input, non_click_frames, non_out, info,
                          **common)
        else:
            print("      (no non-click frames to write)")

    print("[5/5] Done.")


if __name__ == "__main__":
    main()
