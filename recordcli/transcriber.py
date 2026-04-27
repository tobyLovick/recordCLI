from faster_whisper import WhisperModel


def load_model(size="base"):
    print(f"Loading Whisper model '{size}' (downloads on first use)...")
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, audio, context="", beam_size=1, vad_filter=False):
    """Transcribe a numpy float32 mono 16kHz audio array."""
    segments, _ = model.transcribe(audio, beam_size=beam_size, language="en",
                                   initial_prompt=context or None,
                                   vad_filter=vad_filter)
    return " ".join(seg.text.strip() for seg in segments)


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
    return " ".join(parts)
