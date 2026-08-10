import re

from faster_whisper import WhisperModel

# Whisper renders hesitations/trailing-off speech as literal "..." (a
# convention from its training transcripts), not just for genuine long
# silences. Strip runs of 2+ dots (with optional spacing) and "…".
_PAUSE_MARKER_RE = re.compile(r"(?:\s*\.\s*){2,}|…+")


def _strip_pause_markers(text):
    return re.sub(r"\s+", " ", _PAUSE_MARKER_RE.sub(" ", text)).strip()


def load_model(size="base"):
    print(f"Loading Whisper model '{size}' (downloads on first use)...")
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, audio, context="", beam_size=1, vad_filter=False):
    """Transcribe a numpy float32 mono 16kHz audio array.

    Returns (text, reliable). `reliable` is False when faster-whisper's own
    per-segment quality signals (the same ones it uses internally for
    temperature fallback) indicate a hallucinated/degenerate segment —
    callers should avoid feeding such text back in as a future prompt,
    since Whisper's initial_prompt biases decoding toward repeating it.
    """
    segments, _ = model.transcribe(audio, beam_size=beam_size, language="en",
                                   initial_prompt=context or None,
                                   vad_filter=vad_filter)
    segments = list(segments)
    text = _strip_pause_markers(" ".join(seg.text.strip() for seg in segments))
    reliable = all(
        seg.compression_ratio < 2.4 and seg.avg_logprob > -1.0 and seg.no_speech_prob < 0.6
        for seg in segments
    )
    return text, reliable


def transcribe_file(model, path, beam_size=5, vad_filter=True):
    """Transcribe an audio file with a tqdm progress bar driven by segment timestamps."""
    from tqdm import tqdm
    segments, info = model.transcribe(path, beam_size=beam_size, language="en",
                                      vad_filter=vad_filter)
    parts = []
    with tqdm(total=round(info.duration), unit="s", desc="  transcribing",
              bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s") as pbar:
        pos = 0.0
        for seg in segments:
            parts.append(seg.text.strip())
            pbar.update(seg.end - pos)
            pos = seg.end
    return _strip_pause_markers(" ".join(parts))
