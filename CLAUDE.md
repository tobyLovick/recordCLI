# recordCLI — Claude Code notes

## Install & run

```bash
pipx install -e .   # installs `record` command globally
record --liveupdate # start recording
```

After editing source, reinstall with `pipx install . --force`.

## Structure

- `recordcli/cli.py` — argument parsing, recording modes, import flow
- `recordcli/recorder.py` — audio capture, silence detection, chunking
- `recordcli/transcriber.py` — faster-whisper wrapper
- `recordcli/filer.py` — naming, slugifying, saving transcripts
- `transcript-hook.sh` — Claude Code hook (delta-injects live transcript)

## Key defaults

- Device: 8 (PipeWire on this machine — adjust `--device` for others)
- Silence threshold: 0.1 RMS
- Normal mode: medium model, beam=5, vad_filter=True
- Liveupdate mode: small model, beam=1, vad_filter=False
- Max chunk: 30s (forces flush on continuous speech)
- Notes dir: `~/recordCLI/notes/`

## Hook setup (already configured globally)

`~/.claude/settings.json` has a `UserPromptSubmit` hook pointing to `transcript-hook.sh`. It reads byte-delta from `current.txt` and injects only new content per message. Offset resets when recording stops.
