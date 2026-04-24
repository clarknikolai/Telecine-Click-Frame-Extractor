# Telecine-Click-Frame-Extractor
A Python script to use with FFMPEG to extract clean film frames from MovieStuff telecine video captures by detecting audio sync pulses.

Designed for **MovieStuff HD telecine units** (and similar telecine machines that do not have a rotating shutter in the film projector) that embed a click/pulse in the audio track for each static film frame. The script detects these pulses, maps them to video frames, and outputs a new video containing only the actual film frames — removing all inter-frame pull-down artifacts.

## How It Works

**This is a two step process. First capture to a movie file using any capture device. Then process that file with this script to create a new file.

1. **Audio Analysis** — Detects click/pulse events in the audio track using envelope-based rising-edge detection
2. **Frame Mapping** — Maps each audio pulse timestamp to the corresponding video frame
3. **Deinterlacing** — Optionally extracts a single field (top or bottom) to eliminate interlace artifacts
4. **Output** — Produces a clean video at the desired film frame rate (16, 18, or 24 fps)

## Requirements

- **Python 3.7+**
- **NumPy** — `pip install numpy`
- **FFmpeg & FFprobe** — must be installed and available on PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from https://ffmpeg.org/download.html

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/telecine-click-extractor.git
cd telecine-click-extractor
pip install numpy
```


## Usage


### Regular 8mm (16 fps)

```bash
python click_frames.py --input capture.mov --output film.mov --mode video --out-fps 16 --field both --no-smart
```


### Super 8 (18 fps)

```bash
python click_frames.py --input capture.mov --output film.mov --mode video --out-fps 18 --field both --no-smart
```


### 16mm / Standard Film (24 fps)

```bash
python click_frames.py --input capture.mov --output film.mov --mode video --out-fps 24 --field both --no-smart
```

### Keep Interlaced (for 30p/30PsF captures)

If your camera captures in 30 Progressive segmented Frame (30PsF) mode, both fields are identical so you can keep full 1080 vertical resolution:

```bash
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field both --no-smart
```

### Flip Vertically (for upside-down telecine)

Many telecine units produce an upside-down image. Use `--flip-vertical` to flip the output:

```bash
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field both --no-smart --flip-vertical
```

### Process Only a Section (for quick testing)

Test with just the first 30 seconds to quickly check offset settings before processing the whole file:

```bash
python click_frames.py --input capture.mov --output test.mov \
    --mode video --out-fps 18 --field both --no-smart --duration 30
```

Process a 30-second segment starting at 1 minute (to test settings):

```bash
python click_frames.py --input capture.mov --output test.mov \
    --mode video --out-fps 18 --field both --no-smart \
    --start 60 --duration 30
```

### Extract as Numbered Stills (PNG)

```bash
python click_frames.py --input capture.mov --output frames/ --mode images
```

### Adjust Frame Offset

If the clean frame is one frame before or after the click:

```bash
# One frame later
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field bottom --no-smart --offset 1

# One frame earlier
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field bottom --no-smart --offset -1
```

### Smart Frame Selection

Instead of using the exact click frame, analyze nearby frames and pick the sharpest one:

```bash
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field bottom --search-range 2
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input video file (telecine capture) |
| `--output` | (required) | Output video file or folder |
| `--mode` | `video` | `video`, `images`, or `compare` |
| `--out-fps` | source fps | Output frame rate: `16`, `18`, or `24` |
| `--field` | `both` | Deinterlace: `top`, `bottom`, or `both` |
| `--flip-vertical` | off | Flip output vertically (for upside-down telecine) |
| `--start` | `0` | Start time in seconds (for section processing) |
| `--duration` | whole file | Duration in seconds (for section processing) |
| `--offset` | `0` | Frame offset from click: `-1`, `0`, `+1` |
| `--no-smart` | off | Disable sharpness-based frame selection |
| `--search-range` | `2` | Frames to search around each click |
| `--max-score` | auto | Minimum sharpness threshold (0 = disable) |
| `--threshold` | `0.3` | Audio detection sensitivity (0-1) |
| `--min-gap` | `0.02` | Minimum seconds between clicks |
| `--log` | none | Write click timestamps to file |

## Recommended Workflow

### Step 1: Test on a Short Segment

Use `--duration 30` to process only the first 30 seconds and quickly verify the output looks correct:

```bash
python click_frames.py --input capture.mov --output test.mov \
    --mode video --out-fps 18 --field bottom --no-smart --duration 30
```

If the output has artifacts, try different options:
- Try `--field top` instead of `bottom`
- Try `--offset -1` or `--offset 1`
- Add `--flip-vertical` if upside down

### Step 2: Process the Full File

Once the settings are right, run on the full file by removing `--duration`:

```bash
python click_frames.py --input capture.mov --output film.mov \
    --mode video --out-fps 18 --field bottom --no-smart
```

## Resolution Support

The script works with any input resolution. Whether your camera captures 1080p, 4K, or any other resolution, no changes are needed — the output matches the input resolution (halved vertically when using `--field top/bottom`).

## File Size Support

The script uses a streaming architecture and handles files of any size. Tested with files up to **24GB** (long films).

Processing speed (approximate):
- 100MB file: ~2 seconds
- 500MB file: ~10 seconds
- 4GB file: ~1-2 minutes
- 24GB file: ~10-15 minutes

## Verifying Output

A verification script is included to scan output files for any remaining artifacts:

```bash
python verify_output.py output.mov
```

This reports suspicious frames with their timecode so you can check them manually.

## Tested With

- **Telecine:** MovieStuff DV8 Sniper HD (8mm/Super 8)
- **Camera:** Canon HV30 at 30PsF
- **Blackmagic Intensity Extreme capture unit
- **Source format:** ProRes 422 LT, 1920x1080, 29.97fps
- **Film formats:** Super 8 (18fps), Regular 8mm (16fps)

- Should work with the following MovieStuff units: Velocity Box, QuickSilver-FX, DV8 Sniper-Pro, DV8 Sniper, DV8 Sniper-HD, Sniper HD Pro, DV8 Sniper HDL, Sniper-16 Pro, Sniper-16, Sniper-16 HD, 9.5mm Sniper Pro and 9.5mm Sniper HD. These have the Auto-Sync feature that use an audio pulse combined with software.

## Background

This script replicates the functionality of the discontinued **VelocityHD** software that originally shipped with MovieStuff telecine units. VelocityHD would capture to a video file, then process it by keeping only the frames corresponding to the photo-cell sync pulses — removing all transition frames where the film was advancing between positions.

For more information about the telecine hardware:
- [MovieStuff 8mm HD Setup](https://www.moviestuff.tv/8mm_hd_setup.html)
- [VelocityHD Instructions](https://www.moviestuff.tv/velocity_hd_instructions.html)

Created for Clark Nikolai by Khubchand Bansal.

## License

MIT
