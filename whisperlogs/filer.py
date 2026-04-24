import re
import soundfile as sf
from pathlib import Path

# matches: "this note should be called X end name", "name this note X", "call this X", etc.
NAME_PATTERN = re.compile(
    r'(?:'
    r'this\s+(?:note\s+)?should\s+be\s+called'
    r'|(?:please\s+)?(?:name|call|title|label)\s+this\s+(?:note\s+)?(?:as\s+)?'
    r'|this\s+(?:note\s+)?is\s+called'
    r'|save\s+this\s+(?:note\s+)?as'
    r')\s+(.+?)(?:\s+end\s*(?:name|tag|note)|[.!?]|$)',
    re.IGNORECASE
)


def extract_name(transcript, override=None):
    if override:
        return slugify(override)
    match = NAME_PATTERN.search(transcript)
    if not match:
        return None
    return slugify(match.group(1).strip())


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text.strip('-')


def clean_transcript(text):
    # collapse runs of spaced dots (4+ dots) into ellipsis
    text = re.sub(r'(\.\s*){4,}', '... ', text)
    return text


def save_transcript(transcript, name, output_dir, timestamp):
    if name:
        folder = output_dir / name
        filename = f"{timestamp}_{name}.txt"
    else:
        folder = output_dir / "untagged"
        filename = f"{timestamp}.txt"

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(clean_transcript(transcript).strip() + "\n")
    return path


def save_audio(audio, output_path, samplerate=16000):
    wav_path = output_path.with_suffix(".wav")
    sf.write(str(wav_path), audio, samplerate)

    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_wav(str(wav_path))
        sound.export(str(output_path.with_suffix(".mp3")), format="mp3")
        wav_path.unlink()
        return output_path.with_suffix(".mp3")
    except Exception:
        print(f"  (MP3 conversion needs ffmpeg — saved as WAV instead)")
        return wav_path
