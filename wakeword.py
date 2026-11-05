"""'Hey Loki' wake word: listens to the microphone, and when it hears the
phrase, records what you say next and hands it to a callback.

Uses openWakeWord (runs locally on the CPU, a few % of one core). The model
file is wakeword/hey_loki.onnx, trained once with the notebook in wakeword/
(see the README's wake word section). Until that file exists, the ready-made
"hey jarvis" model is used, so you can test everything before training.

How a wake-up goes:
  1. Every 80 ms of audio is scored by the wake-word model.
  2. A voice-activity check (Silero VAD) blocks the score unless someone was
     actually speaking, which cuts false wake-ups from typing, fans and other
     noise. It listens to a *boosted* copy of the audio (AutoGain), because
     on quiet, distant speech it otherwise misses the voice and blocks real
     wake-ups. The wake-word model itself hears the original audio.
  3. A score above THRESHOLD for PATIENCE frames in a row = "Hey Loki".
  4. A short chime plays, and what you say next is recorded until you stop
     talking (END_SILENCE), then passed to on_command().

The microphone follows Windows: if the default mic changes (earphones
plugged in or out) or stops sending audio, the listener reopens it, and if
no mic is available at startup it keeps retrying. While Loki is speaking or
F9 is held, the listener ignores the mic, so Loki can't wake itself up.
"""
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

WAKE_DIR = Path(__file__).parent / "wakeword"
WAKE_MODEL = WAKE_DIR / "hey_loki.onnx"              # your trained model
VERIFIER_MODEL = WAKE_DIR / "hey_loki_verifier.pkl"  # optional: tuned to your voice
PLACEHOLDER_MODEL = "hey_jarvis"                      # used until hey_loki.onnx exists

# ---- Tuning (`python wakeword_tools.py test` shows live scores to tune these) ----
THRESHOLD = 0.5          # wake-word score needed (0-1). Higher = fewer false wake-ups
PATIENCE = 2             # ...for this many 80 ms frames in a row
VAD_THRESHOLD = 0.5      # voice-activity score needed for a wake-up to count (0 = off)
VERIFIER_THRESHOLD = 0.3 # base score at which the voice verifier is consulted
AUTO_GAIN = True         # boost quiet speech for the voice check (helps from across the room)
MAX_GAIN = 10            # at most 10x (+20 dB)

END_SILENCE = 0.9        # seconds of quiet that end your command
NO_SPEECH_TIMEOUT = 5.0  # give up if you don't start talking within this
MAX_COMMAND = 15.0       # longest command recorded
MIC_DEVICE = None        # None = follow Windows' default microphone; or a device index/name

STALL_SECONDS = 2.0      # no audio for this long = mic unplugged/asleep → reopen it
DEVICE_CHECK_SECONDS = 3.0   # how often to check whether Windows' default mic changed
RETRY_SECONDS = 5.0      # retry interval when no microphone can be opened

SAMPLE_RATE = 16000
FRAME = 1280             # 80 ms at 16 kHz — openWakeWord's frame size

# Leading wake phrase that Whisper sometimes includes: "Hey Loki, open Spotify"
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi|okay|ok|yo)[\s,]+)?(?:loki|low[\s-]?k(?:ey|ee|i)|lokey|lokee|jarvis)\b[\s,.!?:;-]*",
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
    # vad_threshold=0: the voice check is done in WakeScorer, on boosted audio
    kwargs = dict(wakeword_models=[source], inference_framework="onnx", vad_threshold=0)
    if WAKE_MODEL.exists() and VERIFIER_MODEL.exists():
        kwargs.update(custom_verifier_models={name: str(VERIFIER_MODEL)},
                      custom_verifier_threshold=VERIFIER_THRESHOLD)
    return Model(**kwargs), name


def _default_mic_id():
    """ID of Windows' current default microphone, or None if unknown
    (no pycaw, not Windows, or no mic)."""
    try:
        from pycaw.pycaw import AudioUtilities
        dev = AudioUtilities.GetMicrophone()
        if dev is None:
            return None
        return dev.GetId() if hasattr(dev, "GetId") else getattr(dev, "id", None)
    except Exception:
        return None


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


def level_db(frame):
    """Loudness of an int16 frame in dBFS (0 = maximum, -90 = silence)."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
    return 20 * np.log10(max(rms, 1.0) / 32768)


class AutoGain:
    """Brings quiet speech up to a steady level — but only sound that stands
    out from the room's background. Steady noise (hiss, a fan) is never
    boosted, or the voice check would start mistaking it for speech."""

    def __init__(self, target=3000, max_gain=MAX_GAIN, ratio=2.0):
        self.target, self.max_gain, self.ratio = target, max_gain, ratio
        self.recent = deque(maxlen=20)       # last 1.6 s: how loud is it right now
        self.history = deque(maxlen=75)      # last 6 s: what's the room's background level
        self.gain = 1.0

    def reset(self):
        self.recent.clear()
        self.history.clear()
        self.gain = 1.0

    def __call__(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) + 1e-9
        self.recent.append(rms)
        self.history.append(rms)
        level, floor = max(self.recent), float(np.percentile(self.history, 20))
        self.gain = 1.0 if level < self.ratio * floor else min(self.max_gain, max(1.0, self.target / level))
        if self.gain == 1.0:
            return frame
        return np.clip(frame.astype(np.float32) * self.gain, -32768, 32767).astype(np.int16)


class WakeScorer:
    """Wake-word score for each 80 ms frame: the model hears the original
    audio; the voice check hears the boosted copy."""

    def __init__(self):
        from openwakeword.vad import VAD
        self.model, self.name = _load_model()
        self.vad = VAD()
        self.gain = AutoGain()
        self.speech = deque(maxlen=10)       # voice-check scores of recent frames

    def reset(self):
        self.model.reset()
        self.vad.reset_states()
        self.gain.reset()
        self.speech.clear()

    def __call__(self, frame):
        """Returns (score, boosted_frame, speech_probability)."""
        boosted = self.gain(frame) if AUTO_GAIN else frame
        score = float(self.model.predict(frame)[self.name])
        speech = float(self.vad.predict(boosted, frame_size=640))
        self.speech.append(speech)
        if VAD_THRESHOLD > 0:
            # The wake word ends ~0.5 s after the speech starts, so check the
            # voice score from 0.4-0.56 s ago (the same window openWakeWord uses)
            window = list(self.speech)[-7:-4]
            if not window or max(window) < VAD_THRESHOLD:
                score = 0.0
        return score, boosted, speech


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
        self._mic_id = None
        self._mic_problem = None                  # last mic error, logged once
        self.error = None

    # ---- lifecycle ----
    def start(self):
        """Start listening. Returns False (and sets .error) only if openWakeWord
        itself can't load; a missing microphone is retried in the background."""
        if self._running:
            return True
        try:
            import sounddevice  # noqa: F401
            self._scorer = WakeScorer()
            from openwakeword.vad import VAD
            self._end_vad = VAD()                   # for detecting the end of your command
        except Exception as e:
            self.error = str(e)
            print(f"[WAKE] couldn't start: {e}", flush=True)
            return False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="wakeword")
        self._thread.start()
        verifier = " + your voice verifier" if (WAKE_MODEL.exists() and VERIFIER_MODEL.exists()) else ""
        print(f"[WAKE] listening for '{self._scorer.name}'{verifier}", flush=True)
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        self._close_stream()
        print("[WAKE] stopped", flush=True)

    @property
    def running(self):
        return self._running

    # ---- microphone ----
    def _open_stream(self):
        """Open the mic (Windows' current default unless MIC_DEVICE is set)."""
        import sounddevice as sd
        self._close_stream()
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
                device=MIC_DEVICE, callback=self._on_audio)
            self._stream.start()
        except Exception as e:
            if str(e) != self._mic_problem:         # don't fill the log while retrying
                print(f"[WAKE] no microphone yet ({e}); retrying every {RETRY_SECONDS:g} s", flush=True)
                self._mic_problem = str(e)
            self._close_stream()
            return False
        self._mic_problem = None
        self._mic_id = _default_mic_id()
        self._drain()
        try:
            name = sd.query_devices(self._stream.device, "input")["name"]
        except Exception:
            name = "default microphone"
        print(f"[WAKE] microphone: {name}", flush=True)
        return True

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

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
        self._scorer.reset()

    # ---- main loop ----
    def _run(self):
        try:                                        # pycaw (COM) needs this on each thread
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        was_busy, above, last_audio, next_check = False, 0, time.monotonic(), 0.0
        while self._running:
            now = time.monotonic()
            if self._stream is None:
                if not self._open_stream():
                    time.sleep(RETRY_SECONDS)
                    continue
                last_audio = time.monotonic()

            if MIC_DEVICE is None and now >= next_check:
                next_check = now + DEVICE_CHECK_SECONDS
                mic = _default_mic_id()
                if mic is not None and mic != self._mic_id:
                    print("[WAKE] Windows' default microphone changed — switching", flush=True)
                    self._close_stream()
                    continue

            frame = self._next_frame()
            if frame is None:
                if time.monotonic() - last_audio > STALL_SECONDS:
                    print("[WAKE] microphone stopped sending audio — reopening", flush=True)
                    self._close_stream()
                continue
            last_audio = time.monotonic()
            if len(frame) != FRAME:
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
            score, _, _ = self._scorer(frame)
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
            last_audio = time.monotonic()           # on_command may have taken a while

    def _record_command(self):
        """Record until you stop talking. Returns float32 audio, or None."""
        self._end_vad.reset_states()
        chunks, started, quiet, elapsed = [], False, 0.0, 0.0
        step = FRAME / SAMPLE_RATE
        while self._running and elapsed < MAX_COMMAND:
            frame = self._next_frame()
            if frame is None:
                break
            elapsed += step
            boosted = self._scorer.gain(frame) if AUTO_GAIN else frame
            chunks.append(boosted)
            speech = self._end_vad.predict(boosted, frame_size=640) > 0.5
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
