import argparse
import os
import signal
import subprocess
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from . import recorder as rec_module
from . import transcriber
from . import filer
from . import tapaudio
from . import trimmer

_PID_FILE = Path.home() / ".recordcli_pid"


def _ping_claude():
    subprocess.run(["notify-send", "-t", "3000", "recordCLI", "Chunk processed"],
                   capture_output=True)

DEFAULT_NOTES_DIR = Path.home() / "recordCLI" / "notes"


def main():
    parser = argparse.ArgumentParser(
        prog="record",
        description="Voice notes — speak, stop with Ctrl+C, auto-filed by TAG: phrase.",
    )
    parser.add_argument("--liveupdate", action="store_true",
                        help="Low-latency mode: small model, beam=1, fast chunking")
    parser.add_argument("--fast", action="store_true",
                        help="Use beam=1 for near real-time decoding, without --liveupdate's "
                             "hook-visible current.txt writes/notifications")
    parser.add_argument("--record", action="store_true",
                        help="Just record to mp3, no transcription — no model load, no CPU "
                             "load while recording. Transcribe later with --transcribe")
    parser.add_argument("--savemp3", action="store_true",
                        help="Save the audio recording alongside the transcript")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model (default: small for --liveupdate/--fast, medium otherwise)")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTES_DIR,
                        help=f"Directory to save notes (default: {DEFAULT_NOTES_DIR})")
    parser.add_argument("--silence", type=float, default=0.1,
                        help="RMS silence threshold (default: 0.1)")
    def device_arg(value):
        if value == tapaudio.DEVICE_NAME:
            return value
        return int(value)

    parser.add_argument("--device", type=device_arg, default=8,
                        help="Audio input device index (run 'record --listdevices' to see options), "
                             f"or '{tapaudio.DEVICE_NAME}' to capture whatever is playing through your speakers "
                             "(e.g. a browser tab or Zoom call) instead of the mic")
    parser.add_argument("--listdevices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--name", type=str, default=None,
                        help="Name/tag for this note (overrides any spoken name)")
    parser.add_argument("--continue", dest="continue_file", nargs="?", const="",
                        help="Append to an existing note (path, or omit to pick interactively)")
    parser.add_argument("--transcribe", type=Path, default=None, metavar="FILE",
                        help="Transcribe an existing audio file (m4a, mp3, wav, etc.)")
    parser.add_argument("--import", dest="do_import", action="store_true",
                        help="Download and transcribe new recordings from Google Drive")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output (useful when run by Claude)")
    args = parser.parse_args()

    if args.listdevices:
        import sounddevice as sd
        print(sd.query_devices())
        print(f"\n'{tapaudio.DEVICE_NAME}': capture desktop/tab audio instead of a mic "
              "(pass --device tabaudio)")
        return

    if args.do_import:
        if args.model is None:
            args.model = "medium"
        args.output.mkdir(parents=True, exist_ok=True)
        _run_import(args, quiet=args.quiet)
        return

    if args.transcribe:
        if args.model is None:
            args.model = "medium"
        model = transcriber.load_model(args.model, quiet=args.quiet)
        if not args.quiet:
            print(f"Transcribing {args.transcribe}...")
        trimmed_path, _ = trimmer.trim_silence_file(args.transcribe, quiet=args.quiet)
        try:
            text = transcriber.transcribe_file(model, str(trimmed_path), quiet=args.quiet)
        finally:
            if trimmed_path != Path(args.transcribe):
                trimmed_path.unlink(missing_ok=True)
        print(text)
        timestamp = datetime.fromtimestamp(args.transcribe.stat().st_mtime).strftime("%Y-%m-%d_%H-%M")
        args.output.mkdir(parents=True, exist_ok=True)
        _file_and_report(text, [], args, timestamp)
        return

    if args.continue_file is not None:
        args.continue_file = _pick_continue_file(args.continue_file, args.output)
        if args.continue_file is None:
            return

    if args.record:
        args.output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        rec = rec_module.AudioRecorder(silence_threshold=args.silence)
        _run_record_only(rec, args, timestamp)
        return

    if args.model is None:
        args.model = "small" if (args.liveupdate or args.fast) else "medium"

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model = transcriber.load_model(args.model)
    rec = rec_module.AudioRecorder(silence_threshold=args.silence)

    print("\nRecording... Press Ctrl+C to stop.\n")

    if args.liveupdate:
        _run_chunked(rec, model, args, timestamp, beam_size=1, context_len=100, live=True, vad_filter=True)
    else:
        beam_size = 1 if args.fast else 5
        _run_chunked(rec, model, args, timestamp, beam_size=beam_size, context_len=200, live=False, vad_filter=True)


def _run_record_only(rec, args, timestamp):
    rec.start(device=args.device)
    print("\nRecording... Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()

    audio = rec.get_all_audio()
    if len(audio) < 1000:
        print("No audio captured.")
        return

    name = args.name
    folder = args.output / (name if name else "untagged")
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{name}" if name else timestamp
    saved = filer.save_audio(audio, folder / f"{stem}.wav")
    print(f"Saved: {saved}")
    print(f"Tip  : transcribe later with 'record --transcribe {saved}'")


def _run_chunked(rec, model, args, timestamp, beam_size, context_len, live, vad_filter=False):
    tmp_path = args.output / f".tmp_{timestamp}.txt"
    current_path = args.output / "current.txt"
    offset_path = Path.home() / ".recordcli_offset"
    audio_chunks = []

    if live:
        current_path.unlink(missing_ok=True)
        offset_path.write_text("0")

    _PID_FILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGUSR1, lambda sig, frame: rec_module.trigger_flush())

    rec.start(device=args.device)
    if live:
        print(f"[live] Writing to {tmp_path}")
    print("-" * 50)

    transcript_so_far = ""
    try:
        for chunk, manual in rec.iter_speech_chunks():
            audio_chunks.append(chunk)
            text, reliable = transcriber.transcribe(model, chunk,
                                          context=transcript_so_far[-context_len:],
                                          beam_size=beam_size,
                                          vad_filter=vad_filter)
            if text.strip():
                line = text.strip() + " "
                if reliable:
                    transcript_so_far += line
                with open(tmp_path, "a") as f:
                    f.write(line)
                if live:
                    with open(current_path, "a") as f:
                        f.write(line)
                print(text.strip(), end=" ", flush=True)
                if live and manual:
                    _ping_claude()
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
        _PID_FILE.unlink(missing_ok=True)
        remaining = rec.get_all_audio()
        if len(remaining) > 1000:
            audio_chunks.append(remaining)
            text, _ = transcriber.transcribe(model, remaining,
                                          context=transcript_so_far[-context_len:],
                                          beam_size=beam_size,
                                          vad_filter=vad_filter)
            if text.strip():
                line = text.strip() + " "
                with open(tmp_path, "a") as f:
                    f.write(line)
                if live:
                    with open(current_path, "a") as f:
                        f.write(line)
                print(text.strip(), end=" ", flush=True)

    if live:
        current_path.unlink(missing_ok=True)
        offset_path.write_text("0")

    print("\n" + "-" * 50)

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        print("No audio captured.")
        if tmp_path.exists():
            tmp_path.unlink()
        return

    transcript = tmp_path.read_text()
    tmp_path.unlink()

    _file_and_report(transcript, audio_chunks, args, timestamp)



def _run_import(args, quiet=False):
    from . import gdrive
    import tempfile

    def log(*a, **kw):
        if not quiet:
            print(*a, **kw)

    log("Connecting to Google Drive...")
    service = gdrive.authenticate()

    files = gdrive.list_new_files(service, folder_name=gdrive.DRIVE_FOLDER)
    if not files:
        log("No new files to import.")
        return

    log(f"Found {len(files)} new file(s).")
    model = transcriber.load_model(args.model, quiet=quiet)

    for i, f in enumerate(files, 1):
        name = f["name"]
        ext = Path(name).suffix.lower()
        if ext not in gdrive.AUDIO_EXTENSIONS:
            log(f"Skipping {name} (not audio)")
            gdrive.mark_imported(f["id"])
            continue

        log(f"\n[{i}/{len(files)}] {name}")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        trimmed_path = tmp_path
        try:
            gdrive.download_file(service, f["id"], tmp_path)
            trimmed_path, _ = trimmer.trim_silence_file(tmp_path, quiet=quiet)
            text = transcriber.transcribe_file(model, str(trimmed_path), quiet=quiet)
        finally:
            tmp_path.unlink(missing_ok=True)
            if trimmed_path != tmp_path:
                trimmed_path.unlink(missing_ok=True)

        if not text.strip():
            log("  No speech detected, skipping.")
            gdrive.mark_imported(f["id"])
            continue

        spoken_name = filer.extract_name(text, override=args.name)
        if spoken_name:
            note_name = spoken_name
            log(f"  Name: {note_name}")
        else:
            if quiet:
                note_name = Path(name).stem
            else:
                preview = " ".join(text.split()[:40])
                print(f"  Preview: {preview}...")
                note_name = input(f"  Name this note (Enter for '{Path(name).stem}'): ").strip() or Path(name).stem
        created = f.get("createdTime", "")
        timestamp = created[:16].replace("T", "_").replace(":", "-") if created else datetime.now().strftime("%Y-%m-%d_%H-%M")
        final_path = filer.save_transcript(text, note_name, args.output, timestamp)
        print(f"Saved: {final_path}")
        gdrive.mark_imported(f["id"])


def _pick_continue_file(path, notes_dir):
    import questionary
    if path:
        p = Path(path)
        if not p.exists():
            print(f"File not found: {path}")
            return None
        return p
    # gather all txt files sorted by modification time (newest first)
    files = sorted(notes_dir.rglob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        print("No existing notes found.")
        return None
    choices = [str(f.relative_to(notes_dir)) for f in files]
    chosen = questionary.select("Which note to continue?", choices=choices).ask()
    if chosen is None:
        return None
    return notes_dir / chosen


def _file_and_report(transcript, audio_chunks, args, timestamp):
    if args.continue_file:
        with open(args.continue_file, "a") as f:
            f.write("\n\n--- continued " + timestamp + " ---\n")
            f.write(transcript.strip() + "\n")
        final_path = args.continue_file
        print(f"\nAppended to: {final_path}")
    else:
        name = filer.extract_name(transcript, override=args.name)
        final_path = filer.save_transcript(transcript, name, args.output, timestamp)
        if name:
            print(f"\nName : {name}")
        else:
            print("\nNo name found — saved to untagged/")
            print("Tip  : say 'name this note X end name' or use --name flag.")

    print(f"Saved: {final_path}")

    if args.savemp3 and audio_chunks:
        audio = np.concatenate(audio_chunks)
        audio_path = filer.save_audio(audio, final_path.with_suffix(".mp3"))
        print(f"Audio: {audio_path}")
