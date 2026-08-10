import queue
import subprocess
import threading
import sounddevice as sd
import numpy as np

from . import tapaudio

SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.5  # seconds before a chunk is yielded

_force_flush = threading.Event()


def trigger_flush():
    _force_flush.set()


class AudioRecorder:
    def __init__(self, silence_threshold=SILENCE_THRESHOLD, silence_duration=SILENCE_DURATION):
        self.samplerate = SAMPLERATE
        self.silence_threshold = silence_threshold
        self.silence_samples = int(silence_duration * SAMPLERATE)
        self._q = queue.Queue()
        self._recording = False
        self._stream = None
        self._tap_proc = None
        self._tap_thread = None

    def _callback(self, indata, frames, time, status):
        self._q.put(indata[:, 0].copy())

    def start(self, device=None):
        self._recording = True
        if device == tapaudio.DEVICE_NAME:
            self._start_tap()
        else:
            self._stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCKSIZE,
                device=device,
                callback=self._callback,
            )
            self._stream.start()

    def _start_tap(self):
        tapaudio.ensure_running()
        self._tap_proc = subprocess.Popen(
            [
                "pw-record",
                "--target", tapaudio.SOURCE_TARGET,
                "--rate", str(SAMPLERATE),
                "--channels", str(CHANNELS),
                "--format", "s16",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        bytes_per_block = BLOCKSIZE * CHANNELS * 2  # s16 = 2 bytes/sample

        def _read_loop():
            proc = self._tap_proc
            while self._recording and proc.poll() is None:
                data = proc.stdout.read(bytes_per_block)
                if not data:
                    break
                pcm16 = np.frombuffer(data, dtype="int16")
                self._q.put((pcm16.astype("float32")) / 32768.0)

        self._tap_thread = threading.Thread(target=_read_loop, daemon=True)
        self._tap_thread.start()

    def stop(self):
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._tap_proc:
            self._tap_proc.terminate()
            self._tap_proc.wait(timeout=2)
            self._tap_proc = None
        if self._tap_thread:
            self._tap_thread.join(timeout=2)
            self._tap_thread = None

    def get_all_audio(self):
        chunks = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")

    def iter_speech_chunks(self, max_chunk_seconds=30):
        """Yield (audio, manual) tuples. manual=True when triggered by SIGUSR1."""
        buffer = []
        silent_samples = 0
        min_chunk_samples = int(0.3 * SAMPLERATE)
        max_chunk_samples = int(max_chunk_seconds * SAMPLERATE)

        while self._recording or not self._q.empty():
            if _force_flush.is_set():
                _force_flush.clear()
                if buffer and sum(len(c) for c in buffer) >= min_chunk_samples:
                    yield np.concatenate(buffer), True
                buffer = []
                silent_samples = 0
                continue

            try:
                chunk = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > self.silence_threshold:
                buffer.append(chunk)
                silent_samples = 0
            else:
                silent_samples += len(chunk)
                if buffer:
                    buffer.append(chunk)
                    if silent_samples >= self.silence_samples:
                        audio = np.concatenate(buffer)
                        if len(audio) >= min_chunk_samples:
                            yield audio, False
                        buffer = []
                        silent_samples = 0
                        continue

            if buffer and sum(len(c) for c in buffer) >= max_chunk_samples:
                yield np.concatenate(buffer), False
                buffer = []
                silent_samples = 0

        if buffer:
            audio = np.concatenate(buffer)
            if len(audio) >= min_chunk_samples:
                yield audio, False
