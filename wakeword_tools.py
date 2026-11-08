"""Helpers for the wake word ("Hey Nexus"). Run from the project folder:

    python wakeword_tools.py setup      one-time: download openWakeWord's base models
    python wakeword_tools.py myvoice    record ~60 clips of you saying the wake phrase, for training
    python wakeword_tools.py noise 20   record 20 min of your room's usual sounds for training
    python wakeword_tools.py test       live meter: see the score as you talk, to tune THRESHOLD
    python wakeword_tools.py verifier   optional: teach the wake word your voice
    python wakeword_tools.py misses     after some use: why the wake word didn't wake

See the README's "Wake word" section for the whole process.
"""
import shutil
import sys
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

import wakeword
from app_info import APP_NAME, WAKE_PHRASE

SAMPLE_RATE = wakeword.SAMPLE_RATE
NOISE_DIR = wakeword.WAKE_DIR / "my_room_noise"
VOICE_DIR = wakeword.WAKE_DIR / "my_voice"
CLIPS_DIR = wakeword.WAKE_DIR / "verifier_clips"


def _save_wav(path, audio_int16):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio_int16.astype(np.int16).tobytes())


def _record(seconds):
    import sounddevice as sd
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype="int16", device=wakeword.MIC_DEVICE)
    sd.wait()
    return audio[:, 0]


# ---------------------------------------------------------------- setup
def setup():
    """openWakeWord's pip package doesn't include its models; fetch them once
    (feature extractor, voice-activity detector, and the "hey jarvis" placeholder)."""
    import openwakeword.utils
    openwakeword.utils.download_models([wakeword.PLACEHOLDER_MODEL])
    wakeword.WAKE_DIR.mkdir(exist_ok=True)
    print("\nDone. Put your trained model at", wakeword.WAKE_MODEL)
    if not wakeword.WAKE_MODEL.exists():
        print(f"Until then, {APP_NAME} answers to \"Hey Jarvis\" so you can try it out.")


# ---------------------------------------------------------------- test
def test():
    """Print the wake-word score live, scored exactly the way the assistant does. Say
    the phrase from different spots in the room, with music or a fan on, and
    see where the score lands. Follows Windows' default mic if you unplug one."""
    import collections
    import sounddevice as sd
    scorer = wakeword.WakeScorer()
    print(f"Listening for '{scorer.name}' (threshold {wakeword.THRESHOLD}, "
          f"{wakeword.PATIENCE} frames in a row). Ctrl+C to stop.")
    print("score = wake word | mic = loudest in the last 1.5 s (talking: about -35 dB or louder) | "
          "voice = voice check, best in the last 1.5 s (needs {:.1f}+)\n".format(wakeword.VAD_THRESHOLD))
    above, peak, peak_at = 0, 0.0, 0.0
    levels, voices = collections.deque(maxlen=19), collections.deque(maxlen=19)
    try:
        while True:                                  # reopen if the mic goes away
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                    blocksize=wakeword.FRAME, device=wakeword.MIC_DEVICE) as stream:
                    try:
                        print("Microphone:", sd.query_devices(stream.device, "input")["name"], "\n")
                    except Exception:
                        pass
                    scorer.reset()
                    while True:
                        frame, _ = stream.read(wakeword.FRAME)
                        frame = frame[:, 0]
                        score, _, speech = scorer(frame)
                        levels.append(wakeword.level_db(frame))
                        voices.append(speech)
                        above = above + 1 if score >= wakeword.THRESHOLD else 0
                        if score > peak or time.time() - peak_at > 3:
                            peak, peak_at = score, time.time()
                        bar = "█" * int(score * 30)
                        woke = "  ← WAKE" if above == wakeword.PATIENCE else ""
                        print(f"\r{score:4.2f} {bar:<30} peak {peak:4.2f} | mic {max(levels):4.0f} dB "
                              f"x{scorer.gain.gain:<4.1f}| voice {max(voices):4.2f}{woke}   ", end="", flush=True)
                        if woke:
                            print()
            except sd.PortAudioError as e:
                print(f"\n\nMicrophone disconnected ({e}). Switching to Windows' default mic...")
                time.sleep(2)
    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------- myvoice
VOICE_ROUNDS = [                      # (how to say it, how many times)
    ("normally, the way you usually would", 15),
    ("quickly, like you're in a hurry", 10),
    ("slowly and clearly", 8),
    ("quietly, almost under your breath", 8),
    ("from farther away: step back a few feet or turn away from the mic", 10),
    ("casually, mid-task, as if you're busy with something else", 9),
]


def _beep():
    import sounddevice as sd
    t = np.arange(int(0.08 * 22050)) / 22050
    sd.play((0.2 * np.sin(2 * np.pi * 990 * t) * np.minimum(1, np.minimum(t, t[::-1]) / 0.01)).astype(np.float32), 22050)
    sd.wait()
    time.sleep(0.1)                   # so the beep itself isn't recorded


def _trim(audio):
    """Cut the silence around the phrase. Training lines every clip up so the
    phrase ends at the end of a ~2 s window, so the end is trimmed tight."""
    win = SAMPLE_RATE // 100                                   # 10 ms
    n = len(audio) // win
    rms = np.sqrt(np.mean(audio[:n * win].astype(np.float32).reshape(n, win) ** 2, axis=1))
    floor = np.percentile(rms, 10)
    loud = np.where(rms > max(rms.max() * 0.08, floor * 3))[0]
    if len(loud) == 0:
        return audio[:0]
    start = max(0, (loud[0] - 10) * win)                       # keep 100 ms before
    end = min(len(audio), (loud[-1] + 5) * win)                # and 50 ms after
    return audio[start:end]


def myvoice():
    """Record you saying the wake phrase ~60 times, in different ways. The
    training notebook mixes these in with its synthetic voices, so the model
    learns how *you* actually say it — the biggest single fix for misses."""
    from openwakeword.vad import VAD
    vad = VAD()
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(VOICE_DIR.glob("*.wav")))
    total = sum(k for _, k in VOICE_ROUNDS)
    print(f"Recording \"{WAKE_PHRASE}\" {total} times in {len(VOICE_ROUNDS)} rounds. "
          f"Say it once after each beep. Use the mic you'll normally use.")
    if n:
        print(f"({n} clips already in {VOICE_DIR}; new ones are added. Delete the folder to start over.)")
    try:
        for r, (how, count) in enumerate(VOICE_ROUNDS, 1):
            input(f"\nRound {r}/{len(VOICE_ROUNDS)}: say it {how}. Press Enter to start...")
            done = 0
            while done < count:
                _beep()
                audio = _record(2.2)
                clip = _trim(audio)
                vad.reset_states()
                speech = max((vad.predict(clip[i:i + 1280], frame_size=640)
                              for i in range(0, len(clip) - 1280, 1280)), default=0)
                if not 0.35 <= len(clip) / SAMPLE_RATE <= 1.9 or speech < 0.5:
                    print("    (didn't catch that — once more)")
                    continue
                _save_wav(VOICE_DIR / f"voice_{n:04d}.wav", clip)
                n, done = n + 1, done + 1
                print(f"\r  {done}/{count}", end="", flush=True)
            print()
    except KeyboardInterrupt:
        print("\nStopped early — the clips so far are kept.")
    zip_path = wakeword.WAKE_DIR / "my_voice.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for wav in sorted(VOICE_DIR.glob("*.wav")):
            z.write(wav, wav.name)
    print(f"\n{n} clips → {zip_path}\nGive that zip to the training notebook (cell 5).")


# ---------------------------------------------------------------- noise
def noise(minutes=20):
    """Record your room's everyday sounds (fan, typing, music or TV you usually
    have on, you talking on calls — anything but the wake word). Training mixes
    these under the synthetic wake-phrase clips, so the model learns to ignore
    exactly the noise it will hear here. Saves 10-second clips + a zip to upload."""
    minutes = float(minutes)
    print(f"Recording {minutes:g} minutes of room sound. Go about your usual work — "
          f"just don't say the wake word.\nCtrl+C to stop early (what's recorded is kept).")
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    start = len(list(NOISE_DIR.glob("*.wav")))
    n_clips = max(1, round(minutes * 6))         # 10-second clips
    try:
        for saved in range(1, n_clips + 1):
            _save_wav(NOISE_DIR / f"room_{start + saved - 1:04d}.wav", _record(10))
            left = (n_clips - saved) / 6
            print(f"\r  {saved}/{n_clips} clips saved, {left:4.1f} min left", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    zip_path = wakeword.WAKE_DIR / "my_room_noise.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for wav in sorted(NOISE_DIR.glob("*.wav")):
            z.write(wav, wav.name)
    total = len(list(NOISE_DIR.glob("*.wav")))
    print(f"{total} clips ({total * 10 / 60:.0f} min) → {zip_path}\n"
          f"Upload that zip to Colab (see the README).")


# ---------------------------------------------------------------- verifier
SENTENCES = [
    "The quick brown fox jumps over the lazy dog, and then it naps in the sun.",
    "I need to finish the measurement before lunch, then look at the data again.",
    "Could you set a reminder for tomorrow and open my notes from last week?",
]


def verifier():
    """Record a few clips of you, then train a small model that only lets
    wake-ups through when it's your voice. Cuts false wake-ups from TV, music
    and other people a lot — at the cost of not answering anyone else."""
    if not wakeword.WAKE_MODEL.exists():
        sys.exit(f"Train your wake word first: {wakeword.WAKE_MODEL} doesn't exist yet.")
    import openwakeword
    if CLIPS_DIR.exists():
        shutil.rmtree(CLIPS_DIR)
    positives, negatives = [], []

    print(f"Part 1 of 2: say \"{WAKE_PHRASE}\" when prompted, the way you normally would.")
    for i in range(6):
        input(f"  [{i + 1}/6] Press Enter, then say \"{WAKE_PHRASE}\"...")
        path = CLIPS_DIR / f"positive_{i}.wav"
        _save_wav(path, _record(2.5))
        positives.append(str(path))

    print("\nPart 2 of 2: read each sentence aloud at a normal pace.")
    for i, sentence in enumerate(SENTENCES):
        input(f"  [{i + 1}/{len(SENTENCES)}] Press Enter, then read:\n      \"{sentence}\"")
        path = CLIPS_DIR / f"negative_{i}.wav"
        _save_wav(path, _record(7))
        negatives.append(str(path))
    noise_clips = sorted(NOISE_DIR.glob("*.wav"))[:6]      # your room, if recorded
    negatives += [str(p) for p in noise_clips]

    print("\nTraining your voice verifier...")
    try:
        openwakeword.train_custom_verifier(
            positive_reference_clips=positives,
            negative_reference_clips=negatives,
            output_path=str(wakeword.VERIFIER_MODEL),
            model_name=str(wakeword.WAKE_MODEL),
            inference_framework="onnx",
        )
    except ValueError as e:
        sys.exit(f"\nCouldn't train: {e}\nThe wake-word model didn't detect your \"{WAKE_PHRASE}\" "
                 f"clips. Check with 'python wakeword_tools.py test' first.")
    print(f"\nSaved {wakeword.VERIFIER_MODEL}. Restart {APP_NAME} to use it. "
          f"Delete that file to go back to answering anyone.")


# ---------------------------------------------------------------- misses
def misses():
    """Summarize wake-ups and near misses from the assistant's logs (logs/log_*.txt)."""
    import re
    from collections import Counter
    wakes, near, old_wakes = [], [], 0
    for log in sorted((Path(__file__).parent / "logs").glob("log_*.txt")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if m := re.search(r"\[WAKE\] heard wake word \(score [\d.]+, peak ([\d.]+)", line):
                wakes.append(float(m.group(1)))
            elif "[WAKE] heard wake word" in line:
                old_wakes += 1                       # from before near misses were logged
            elif m := re.search(r"\[WAKE\] near miss: peak ([\d.]+) — (.+)", line):
                near.append((float(m.group(1)), m.group(2)))
    if not wakes and not near:
        sys.exit("No wake-word activity logged yet (or only from before near-miss logging). "
                 f"Use {APP_NAME} for a while in tray mode, then run this again.")
    print(f"Since near-miss logging started:  wake-ups {len(wakes)}   near misses {len(near)}")
    if old_wakes:
        print(f"(plus {old_wakes} earlier wake-ups, not counted: there's no record of misses from then)")
    print()
    if wakes:
        print(f"Wake-up peak scores: median {np.median(wakes):.2f}, lowest {min(wakes):.2f}")
    if not near:
        print("No near misses. If it still feels like it misses you, those attempts scored under "
              f"{wakeword.NEAR_MISS} (not logged): the model isn't reacting to the way you say it, "
              "and retraining with your own voice is the fix.")
        return
    reasons = Counter("score too low" if r.startswith("score below") else
                      "voice check" if "voice check" in r else "too brief" for _, r in near)
    print("Near misses by reason:")
    for reason, n in reasons.most_common():
        print(f"  {reason:14} {n}")
    low = sorted(p for p, r in near if r.startswith("score below"))
    if low:
        buckets = Counter(min(int(p * 10), 9) for p in low)
        print("\nPeak scores of the 'too low' misses:")
        for b in range(2, 10):
            if buckets[b]:
                print(f"  {b / 10:.1f}-{(b + 1) / 10:.1f}  {'█' * buckets[b]} {buckets[b]}")
    print()
    if len(near) < 8:
        print(f"Only {len(near)} near misses so far — too few to recommend a change. "
              f"Keep using it and run this again in a day or two.")
    else:
        top = reasons.most_common(1)[0][0]
        if top == "score too low":
            n_close = sum(1 for p in low if p >= wakeword.THRESHOLD - 0.15)
            if n_close >= len(low) / 2:
                print(f"Most misses landed just under the threshold: try THRESHOLD = "
                      f"{wakeword.THRESHOLD - 0.1:.1f} in wakeword.py.")
            else:
                print("Most misses scored well under the threshold: the model doesn't recognize the way "
                      "you say it. Retraining with recordings of your own voice fixes this best.")
        elif top == "voice check":
            print("The voice check is blocking real attempts: try VAD_THRESHOLD = 0.3 in wakeword.py.")
        else:
            print("Attempts reached the threshold only briefly: try PATIENCE = 1 in wakeword.py.")
    print(f"\n(Near misses can include ordinary speech that happened to score above {wakeword.NEAR_MISS}, "
          f"and attempts under {wakeword.NEAR_MISS} aren't logged at all; the counts are a guide.)")


if __name__ == "__main__":
    commands = {"setup": setup, "myvoice": myvoice, "test": test, "noise": noise,
                "verifier": verifier, "misses": misses}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(1)
    commands[sys.argv[1]](*sys.argv[2:])
