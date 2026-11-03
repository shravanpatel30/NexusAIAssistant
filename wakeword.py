"""'Hey Loki' wake word: listens to the microphone, and when it hears the
phrase, records what you say next and hands it to a callback.

Uses openWakeWord (runs locally on the CPU, a few % of one core). The model
file is wakeword/hey_loki.onnx, trained once in Google Colab (see the
README's "Wake word" section). Until that file exists, the ready-made
"hey jarvis" model is used, so you can test everything before training.

How a wake-up goes:
  1. Every 80 ms of audio is scored by the wake-word model. Silero VAD
     (built into openWakeWord) blocks scores when no one is speaking, which
     cuts false wake-ups from music, typing and other noise.
  2. A score above THRESHOLD for PATIENCE frames in a row = "Hey Loki".
  3. A short chime plays, and what you say next is recorded until you stop
     talking (END_SILENCE), then passed to on_command().
While Loki is speaking or F9 is held, the listener ignores the mic, so Loki
can't wake itself up.
"""
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

WAKE_DIR = Path(__file__).parent / "wakeword"
WAKE_MODEL = WAKE_DIR / "hey_loki.onnx"              # your trained model
VERIFIER_MODEL = WAKE_DIR / "hey_loki_verifier.pkl"  # optional: tuned to your voice
PLACEHOLDER_MODEL = "hey_jarvis"                      # used until hey_loki.onnx exists

# ---- Tuning (wakeword_tools.py test shows live scores to tune these) ----
THRESHOLD = 0.5          # wake-word score needed (0-1). Higher = fewer false wake-ups
PATIENCE = 2             # ...for this many 80 ms frames in a row
VAD_THRESHOLD = 0.5      # Silero speech score needed for a wake-up to count (0 = off)
VERIFIER_THRESHOLD = 0.3 # base score at which the voice verifier is consulted

END_SILENCE = 0.9        # seconds of quiet that end your command
NO_SPEECH_TIMEOUT = 5.0  # give up if you don't start talking within this
MAX_COMMAND = 15.0       # longest command recorded
MIC_DEVICE = None        # None = Windows default microphone; or a device index/name

SAMPLE_RATE = 16000
FRAME = 1280             # 80 ms at 16 kHz — openWakeWord's frame size

# Leading wake phrase that Whisper sometimes includes: "Hey Loki, open Spotify"
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi|okay|ok|yo)[\s,]+)?(?:loki|low[\s-]?key|lokey|jarvis)\b[\s,.!?:;-]*",
    re.IGNORECASE)


def strip_wake_phrase(text):
    return _WAKE_PREFIX_RE.sub("", text).strip()


def model_description():
    """Which phrase to say, for status messages."""
    return "Hey Loki" if WAKE_MODEL.exists() else "Hey Jarvis (placeholder until wakeword/hey_loki.onnx exists)"


def _load_model():
    from openwakeword.model import Model   # imported here so Loki runs without it

    if WAKE_MODEL.exists():
        source, name = str(WAKE_MODEL), WAKE_MODEL.stem
    else:
        source, name = PLACEHOLDER_MODEL, PLACEHOLDER_MODEL
    kwargs = dict(wakeword_models=[source], inference_framework="onnx",
                  vad_threshold=VAD_THRESHOLD)
    if WAKE_MODEL.exists() and VERIFIER_MODEL.exists():
        kwargs.update(custom_verifier_models={name: str(VERIFIER_MODEL)},
                      custom_verifier_threshold=VERIFIER_THRESHOLD)
    return Model(**kwargs), name


def _chime():
    """Two soft rising tones: "I'm listening"."""
    try:
        import sounddevice as sd
        t = np.arange(int(0.07 * 22050)) / 22050
        fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        tone = lambda f: 0.15 * np.sin(2 * np.pi * f * t) * fade
        sd.play(np.concatenate([tone(660), tone(880)]).astype(np.float32), 22050)
    except Exception:
        pass


class WakeWordListener:
    """Background thread: wake word → record command → on_command(audio).

    on_wake():          called right after the wake word (e.g. show "Listening...")
    on_command(audio):  float32 16 kHz numpy array of what you said, or None if
                        you said nothing. Runs on the listener thread; the mic
                        is ignored until it returns, so it can take its time.
    is_busy():          return True while Loki is speaking or recording (F9).
    """

    def __init__(self, on_command, on_wake=None, is_busy=lambda: False):
        self.on_command = on_command
        self.on_wake = on_wake or (lambda: None)
        self.is_busy = is_busy
        self._frames = queue.Queue(maxsize=200)   # ~16 s of audio
        self._running = False
        self._thread = None
        self._stream = None
        self.error = None

    # ---- lifecycle ----
    def start(self):
        """Start listening. Returns False (and sets .error) if it can't."""
        if self._running:
            return True
        try:
            import sounddevice as sd
            self._model, self._name = _load_model()
            from openwakeword.vad import VAD
            self._vad = VAD()                       # for detecting the end of your command
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
                device=MIC_DEVICE, callback=self._on_audio)
            self._stream.start()
        except Exception as e:
            self.error = str(e)
            self._close_stream()
            print(f"[WAKE] couldn't start: {e}", flush=True)
            return False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="wakeword")
        self._thread.start()
        verifier = " + your voice verifier" if (WAKE_MODEL.exists() and VERIFIER_MODEL.exists()) else ""
        print(f"[WAKE] listening for '{self._name}'{verifier}", flush=True)
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_stream()
        print("[WAKE] stopped", flush=True)

    @property
    def running(self):
        return self._running

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ---- audio ----
    def _on_audio(self, indata, frames, time_info, status):
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass                                    # we're busy handling a command

    def _next_frame(self):
        try:
            return self._frames.get(timeout=0.5)
        except queue.Empty:
            return None

    def _drain(self):
        """Forget audio queued up while we were busy, and the model's memory of it."""
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        self._model.reset()

    # ---- main loop ----
    def _run(self):
        was_busy, above = False, 0
        while self._running:
            frame = self._next_frame()
            if frame is None or len(frame) != FRAME:
                continue
            if self.is_busy():                      # Loki talking / F9 held
                was_busy = True
                continue
            if was_busy:
                was_busy, above = False, 0
                self._drain()
                continue

            # Count frames in a row above the threshold ourselves: openWakeWord's
            # own patience= option never fires (it checks scores it has zeroed).
            score = self._model.predict(frame)[self._name]
            above = above + 1 if score >= THRESHOLD else 0
            if above < PATIENCE:
                continue
            above = 0

            print(f"[WAKE] heard wake word (score {score:.2f})", flush=True)
            _chime()
            try:
                self.on_wake()
            except Exception as e:
                print(f"[WAKE] on_wake error: {e}", flush=True)
            audio = self._record_command()
            try:
                self.on_command(audio)
            except Exception as e:
                print(f"[WAKE] command error: {e}", flush=True)
            self._drain()

    def _record_command(self):
        """Record until you stop talking. Returns float32 audio, or None."""
        self._vad.reset_states()
        chunks, started, quiet, elapsed = [], False, 0.0, 0.0
        step = FRAME / SAMPLE_RATE
        while self._running and elapsed < MAX_COMMAND:
            frame = self._next_frame()
            if frame is None:
                break
            elapsed += step
            chunks.append(frame)
            speech = self._vad.predict(frame, frame_size=640) > 0.5
            if speech:
                started, quiet = True, 0.0
            elif started:
                quiet += step
                if quiet >= END_SILENCE:
                    break
            elif elapsed >= NO_SPEECH_TIMEOUT:
                return None
        if not started:
            return None
        return np.concatenate(chunks).astype(np.float32) / 32768.0
