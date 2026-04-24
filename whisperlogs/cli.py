import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from . import recorder as rec_module
from . import transcriber
from . import filer

DEFAULT_NOTES_DIR = Path.home() / "WhisperLogs" / "notes"


def main():
    parser = argparse.ArgumentParser(
        prog="record",
        description="Voice notes — speak, stop with Ctrl+C, auto-filed by TAG: phrase.",
    )
    parser.add_argument("--liveupdate", action="store_true",
                        help="Low-latency mode: small model, beam=1, fast chunking")
    parser.add_argument("--savemp3", action="store_true",
                        help="Save the audio recording alongside the transcript")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model (default: small for --liveupdate, medium otherwise)")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTES_DIR,
                        help=f"Directory to save notes (default: {DEFAULT_NOTES_DIR})")
    parser.add_argument("--silence", type=float, default=0.1,
                        help="RMS silence threshold (default: 0.1)")
    parser.add_argument("--device", type=int, default=8,
                        help="Audio input device index (run 'record --listdevices' to see options)")
    parser.add_argument("--listdevices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--name", type=str, default=None,
                        help="Name/tag for this note (overrides any spoken name)")
    parser.add_argument("--continue", dest="continue_file", nargs="?", const="",
                        help="Append to an existing note (path, or omit to pick interactively)")
    parser.add_argument("--import", dest="do_import", action="store_true",
                        help="Import new recordings from Google Drive")
    parser.add_argument("--transcribe", type=Path, default=None, metavar="FILE",
                        help="Transcribe an existing audio file (m4a, mp3, wav, etc.)")
    args = parser.parse_args()

    if args.listdevices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if args.transcribe:
        if args.model is None:
            args.model = "medium"
        model = transcriber.load_model(args.model)
        print(f"Transcribing {args.transcribe}...")
        text = transcriber.transcribe(model, str(args.transcribe), beam_size=5, vad_filter=True)
        print(text)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        args.output.mkdir(parents=True, exist_ok=True)
        _file_and_report(text, [], args, timestamp)
        return

    if args.do_import:
        try:
            from . import gdrive as _gdrive_check  # noqa: F401
        except ImportError:
            print("--import is not available in this installation.")
            return
        _run_import(args)
        return

    if args.continue_file is not None:
        args.continue_file = _pick_continue_file(args.continue_file, args.output)
        if args.continue_file is None:
            return

    if args.model is None:
        args.model = "small" if args.liveupdate else "medium"

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model = transcriber.load_model(args.model)
    rec = rec_module.AudioRecorder(silence_threshold=args.silence)

    print("\nRecording... Press Ctrl+C to stop.\n")

    if args.liveupdate:
        _run_chunked(rec, model, args, timestamp, beam_size=1, context_len=100, live=True, vad_filter=False)
    else:
        _run_chunked(rec, model, args, timestamp, beam_size=5, context_len=200, live=False, vad_filter=True)


def _run_chunked(rec, model, args, timestamp, beam_size, context_len, live, vad_filter=False):
    tmp_path = args.output / f".tmp_{timestamp}.txt"
    current_path = args.output / "current.txt"
    offset_path = Path.home() / ".whisperlogs_offset"
    audio_chunks = []

    current_path.unlink(missing_ok=True)
    offset_path.write_text("0")

    rec.start(device=args.device)
    if live:
        print(f"[live] Writing to {tmp_path}")
    print("-" * 50)

    transcript_so_far = ""
    try:
        for chunk in rec.iter_speech_chunks():
            audio_chunks.append(chunk)
            text = transcriber.transcribe(model, chunk,
                                          context=transcript_so_far[-context_len:],
                                          beam_size=beam_size,
                                          vad_filter=vad_filter)
            if text.strip():
                line = text.strip() + " "
                transcript_so_far += line
                with open(tmp_path, "a") as f:
                    f.write(line)
                with open(current_path, "a") as f:
                    f.write(line)
                print(text.strip(), end=" ", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
        remaining = rec.get_all_audio()
        if len(remaining) > 1000:
            text = transcriber.transcribe(model, remaining,
                                          context=transcript_so_far[-context_len:],
                                          beam_size=beam_size,
                                          vad_filter=vad_filter)
            if text.strip():
                line = text.strip() + " "
                with open(tmp_path, "a") as f:
                    f.write(line)
                with open(current_path, "a") as f:
                    f.write(line)
                print(text.strip(), end=" ", flush=True)

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



def _run_import(args):
    import questionary
    import tempfile
    from . import gdrive

    print("Connecting to Google Drive...")
    try:
        service = gdrive.authenticate()
    except FileNotFoundError as e:
        print(e)
        return

    files = gdrive.list_new_files(service, folder_name=gdrive.DRIVE_FOLDER)
    if not files:
        print("No new files to import.")
        return

    print(f"Found {len(files)} new file(s).\n")

    if args.model is None:
        args.model = "medium"
    model = None  # load lazily — only if audio files need transcribing

    for f in files:
        name = f["name"]
        suffix = Path(name).suffix.lower()
        print(f"─── {name} ───")

        if suffix in gdrive.TEXT_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            gdrive.download_file(service, f["id"], tmp_path)
            transcript = tmp_path.read_text()
            tmp_path.unlink()
            # show preview
            lines = [l for l in transcript.splitlines() if l.strip()][:3]
            print("\n".join(lines))

        elif suffix in gdrive.AUDIO_EXTENSIONS:
            if model is None:
                model = transcriber.load_model(args.model)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            print("Downloading audio...")
            gdrive.download_file(service, f["id"], tmp_path)
            print("Transcribing...")
            transcript = transcriber.transcribe(model, str(tmp_path), beam_size=5)
            tmp_path.unlink()
            lines = [l for l in transcript.splitlines() if l.strip()][:3]
            print("\n".join(lines))
        else:
            print(f"  Skipping (unsupported type: {suffix})")
            continue

        print()
        note_name = questionary.text("Name this note (blank to skip):").ask()
        if note_name is None:
            break
        if not note_name.strip():
            print("  Skipped.\n")
            gdrive.mark_imported(f["id"])
            continue

        timestamp = f["createdTime"][:16].replace("T", "_").replace(":", "-")
        saved = filer.save_transcript(
            filer.clean_transcript(transcript),
            filer.slugify(note_name),
            args.output,
            timestamp
        )
        gdrive.mark_imported(f["id"])
        print(f"  Saved → {saved}\n")


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
