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
