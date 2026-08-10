"""Strip long stretches of exact digital silence from an audio file before
transcription. Only removes runs of literal all-zero samples — real-world
audio (even a quiet room) always has some non-zero noise floor, so this
never touches genuine quiet speech, only true dead air (e.g. forgetting to
stop a recording).
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLERATE = 16000
MIN_SILENCE_SECONDS = 5.0


def _decode_to_pcm(path):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLERATE),
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype="<i2")


def find_silence_runs(samples, samplerate=SAMPLERATE, min_duration=MIN_SILENCE_SECONDS):
    """Return [(start_sample, end_sample), ...] for runs of exact all-zero samples
    at least `min_duration` seconds long."""
    zero_mask = (samples == 0)
    if not zero_mask.any():
        return []

    diff = np.diff(zero_mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if zero_mask[0]:
        starts.insert(0, 0)
    if zero_mask[-1]:
        ends.append(len(zero_mask))

    min_samples = int(min_duration * samplerate)
    return [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_samples]


def trim_silence_file(path, min_duration=MIN_SILENCE_SECONDS, quiet=False):
    """Strip long exact-silence runs from an audio file.

    Returns (output_path, runs) where output_path is a new temp wav file
    with the silent spans removed (or the original path unchanged if no
    silence was found), and runs is the list of (start_sec, end_sec)
    spans that were cut, in the *original* file's timeline.
    """
    samples = _decode_to_pcm(path)
    runs = find_silence_runs(samples, min_duration=min_duration)
    if not runs:
        return Path(path), []

    keep_mask = np.ones(len(samples), dtype=bool)
    for s, e in runs:
        keep_mask[s:e] = False
    trimmed = samples[keep_mask]

    if not quiet:
        for s, e in runs:
            print(f"  Trimming silence: {s / SAMPLERATE:.0f}s - {e / SAMPLERATE:.0f}s "
                  f"({(e - s) / SAMPLERATE:.0f}s removed)")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    sf.write(str(tmp_path), trimmed, SAMPLERATE)

    runs_seconds = [(s / SAMPLERATE, e / SAMPLERATE) for s, e in runs]
    return tmp_path, runs_seconds
