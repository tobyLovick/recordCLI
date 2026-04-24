# WhisperLogs

A terminal voice notes tool. Speak, stop, get a tagged text file. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```
record --liveupdate
```

## What it does

- Records from your microphone and transcribes with Whisper
- Files notes automatically by a spoken or typed name
- In `--liveupdate` mode, transcribes chunk-by-chunk as you speak — useful for live context with AI assistants
- Detects silence to avoid sending empty audio to Whisper
- Handles continuous speech by flushing every 30 seconds

## Requirements

**System:**
```bash
sudo apt-get install libportaudio2 portaudio19-dev ffmpeg
```
ffmpeg is only needed if you use `--savemp3`.

**Python:** 3.9+

## Installation

Install [pipx](https://pipx.pypa.io) if you don't have it — it makes the `record` command available everywhere without polluting your system Python:

```bash
pip install pipx
pipx ensurepath   # adds pipx bin dir to PATH, then restart your terminal
```

Then install WhisperLogs:

```bash
git clone https://github.com/lovicktoby/recordCLI
cd recordCLI
pipx install -e .
```

The `-e` (editable) flag means changes to the source are picked up immediately without reinstalling — useful if you want to tweak the code.

`record` is now available in any terminal, in any directory.

Whisper model weights download automatically on first use and cache in `~/.cache/huggingface/`.

## Usage

```bash
# record, transcribe on stop (medium model, best accuracy)
record

# live transcription as you speak (small model, low latency)
record --liveupdate

# name the note up front
record --name chapter-3-notes

# keep the audio file
record --savemp3

# continue an existing note (arrow-key picker)
record --continue

# continue a specific file
record --continue notes/my-topic/2026-04-24_my-topic.txt

# choose model explicitly
record --model large

# list audio devices if the default doesn't work
record --listdevices
record --device 4
```

## Naming notes

Say anywhere in your recording:

> *"name this note chapter three end name"*

or

> *"this should be called kepler orbits end name"*

The note gets filed under `~/WhisperLogs/notes/chapter-three/`. If no name is found, it goes to `~/WhisperLogs/notes/untagged/`.

The `--name` flag overrides any spoken name.

## Modes

| | `record` | `record --liveupdate` |
|---|---|---|
| Model default | `medium` | `small` |
| Transcription | on silence / every 30s | on silence / every 30s |
| VAD filter | yes (skips silent chunks) | no (lower latency) |
| Beam size | 5 | 1 |
| Best for | longer notes, accuracy | live context, meetings |

## Live context with Claude Code

In `--liveupdate` mode, transcription is written to `~/WhisperLogs/notes/current.txt` in real time. A Claude Code hook can inject new content into your conversation automatically whenever you send a message.

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/whisperlogs/transcript-hook.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

Then run `record --liveupdate` in one terminal and Claude Code in another. Claude sees what you say as you say it, without you having to paste anything.

## Calibrating the silence threshold

Background noise varies by environment. The default threshold (`--silence 0.1`) works for a quiet room. If transcription triggers on background noise, raise it; if speech isn't being detected, lower it.

Test your noise floor:
```bash
python3 -c "
import sounddevice as sd, numpy as np
audio = sd.rec(int(3*16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
print(f'Noise RMS: {float(np.sqrt(np.mean(audio**2))):.4f}')
"
```

Set `--silence` to roughly 3× your noise RMS.

## Notes structure

```
~/WhisperLogs/notes/
├── my-topic/
│   └── 2026-04-24_14-32_my-topic.txt
├── another-topic/
│   └── 2026-04-24_15-00_another-topic.txt
└── untagged/
    └── 2026-04-24_16-00.txt
```
