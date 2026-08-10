"""Capture desktop/tab audio (whatever is playing through the default output)
without touching the system's default sink or source.

Works by keeping a small pw-loopback process running that taps the
*monitor* of the current default sink (via the @DEFAULT_SINK@ alias, so it
follows you if you switch output devices) and exposes it as a PipeWire
source named "recordcli-tabmic". Recording then happens via `pw-record`
against that source's exact node name.
"""

import json
import shutil
import subprocess
import time

NODE_NAME = "recordcli-tabmic"
SOURCE_TARGET = f"output.{NODE_NAME}"

DEVICE_NAME = "tabaudio"


def _require_pipewire_tools():
    missing = [t for t in ("pw-loopback", "pw-record", "pw-dump") if not shutil.which(t)]
    if missing:
        raise RuntimeError(
            f"Missing PipeWire tools: {', '.join(missing)}. "
            "Tab-audio capture requires PipeWire (pw-loopback, pw-record, pw-dump)."
        )


def is_running():
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    try:
        nodes = json.loads(out)
    except json.JSONDecodeError:
        return False
    for obj in nodes:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        if obj.get("info", {}).get("props", {}).get("node.name") == SOURCE_TARGET:
            return True
    return False


def ensure_running():
    """Start the tap if it isn't already running. Never touches PipeWire defaults."""
    _require_pipewire_tools()
    if is_running():
        return

    subprocess.Popen(
        [
            "pw-loopback",
            "-C", "@DEFAULT_SINK@",
            "--capture-props", "stream.capture.sink=true",
            "--playback-props", f"media.class=Audio/Source node.description={NODE_NAME}",
            "-n", NODE_NAME,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for _ in range(20):
        if is_running():
            return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for the tab-audio tap to start.")
