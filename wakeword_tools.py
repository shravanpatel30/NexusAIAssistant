"""Helpers for the wake word ("Hey Nexus"). Run from the project folder:

    python wakeword_tools.py setup      one-time: download openWakeWord's base models
    python wakeword_tools.py myvoice    record ~75 clips of you saying the wake phrase, for training
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
# Two parts: your usual mic (earphones), then the laptop's built-in mic.
VOICE_PARTS = [
    ("your usual microphone (earphones)", [
        ("normally, the way you usually would", 15),
        ("quickly, like you're in a hurry", 10),
        ("slowly and clearly", 8),
        ("quietly, almost under your breath", 8),
        ("casually, mid-task, as if you're busy with something else", 9),
    ]),
    ("the laptop's built-in microphone (unplug the earphones now)", [
        ("normally, sitting at the laptop", 10),
        ("from 3-6 feet away (step back from the laptop)", 10),
        ("from across the room, a bit louder", 5),
    ]),
]
REC_RATE = 48000              # record at the mic's usual rate, then convert properly to 16 kHz
BLOCK = REC_RATE * 80 // 1000 # 80 ms


def _refresh_devices():
    """PortAudio remembers the devices from when it started; reload the list
    so a mic plugged in or out since then is seen correctly."""
    import sounddevice as sd
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def _pick_mic():
    """Show the input devices; Enter keeps Windows' default. Returns a device or None."""
    import sounddevice as sd
    _refresh_devices()
    try:
        default = sd.query_devices(kind="input")["name"]
    except Exception:
        default = "(none found)"
    print(f"  Microphone: {default}")
    choice = input("  Enter to use it, or L to list other microphones: ").strip().lower()
    if choice != "l":
        return None
    mics = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and sd.query_hostapis(d["hostapi"])["name"] == "MME"]
    for n, (i, d) in enumerate(mics):
        print(f"    {n}: {d['name']}")
    pick = input("  Number (Enter = default): ").strip()
    return mics[int(pick)][0] if pick.isdigit() and int(pick) < len(mics) else None


def _clean_clip(raw48):
    """48 kHz recording of one phrase → tidy 16 kHz clip: proper resampling,
    trimmed to the speech (keeping quiet sounds like a final "s"), short fades."""
    from scipy.signal import resample_poly
    clip = resample_poly(raw48.astype(np.float32), 1, 3)
    win = SAMPLE_RATE // 100                                   # 10 ms
    n = len(clip) // win
    if n < 10:
        return clip[:0].astype(np.int16)
    rms = np.sqrt(np.mean(clip[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-3
    floor = np.percentile(rms, 10)
    sound = np.where(rms > max(floor * 4, rms.max() * 0.01))[0]  # relative to the background, not the peak
    if len(sound) == 0:
        return clip[:0].astype(np.int16)
    start = max(0, (sound[0] - 12) * win)                      # keep 120 ms before
    end = min(len(clip), (sound[-1] + 10) * win)               # and 100 ms after
    clip = clip[start:end]
    fade = min(len(clip) // 4, SAMPLE_RATE // 100)
    ramp = np.linspace(0, 1, fade)
    clip[:fade] *= ramp
    clip[-fade:] *= ramp[::-1]
    return np.clip(clip, -32768, 32767).astype(np.int16)


def _record_round(count, device, vad):
    """Keep the mic open (like a normal recorder — headset mics need a moment
    to settle) and cut out each phrase as you say it. Returns the clips."""
    import queue as _queue
    import sounddevice as sd
    from scipy.signal import resample_poly
    blocks = _queue.Queue()
    clips, rejected = [], 0
    pre = 4                                                    # blocks kept from before speech starts
    with sd.InputStream(samplerate=REC_RATE, channels=1, dtype="int16", blocksize=BLOCK,
                        device=device, callback=lambda d, f, t, s: blocks.put(d[:, 0].copy())):
        time.sleep(0.5)                                        # let the mic settle
        while not blocks.empty():
            blocks.get()
        history, utterance, quiet, idle, skipping = [], None, 0, 0, False
        print(f"  ▶ Say it now  0/{count}", end="", flush=True)
        while len(clips) < count:
            block = blocks.get()
            speech = vad.predict(resample_poly(block.astype(np.float32), 1, 3).astype(np.int16),
                                 frame_size=640) > 0.5
            if skipping:                                       # rest of something too long:
                quiet = 0 if speech else quiet + 1             # wait for a pause
                skipping = quiet < 6
                continue
            if utterance is None:
                history = (history + [block])[-pre:]
                if speech:
                    utterance, quiet = list(history), 0
                else:
                    idle += 1
                    if idle == 125:                            # 10 s of nothing
                        print(f"\r  ▶ Still listening — say it now  {len(clips)}/{count}   ", end="", flush=True)
                continue
            utterance.append(block)
            quiet = 0 if speech else quiet + 1
            if quiet < 6 and len(utterance) < 40:              # until 0.5 s of quiet (max ~3 s)
                continue
            too_long = len(utterance) >= 40
            clip = _clean_clip(np.concatenate(utterance))
            utterance, history, idle = None, [], 0
            secs = len(clip) / SAMPLE_RATE - 0.22                  # the speech itself, without the margins
            if too_long or not 0.45 <= secs <= 2.0:
                rejected += 1
                skipping, quiet = too_long, 0
                hint = "too short — say the whole phrase" if secs < 0.45 and not too_long else "too long — just the phrase, then pause"
                print(f"\r  ({hint})  ▶ again  {len(clips)}/{count}      ", end="", flush=True)
                continue
            clips.append(clip)
            print(f"\r  ▶ Say it now  {len(clips)}/{count}" + " " * 30, end="", flush=True)
    print()
    return clips


def myvoice():
    """Record you saying the wake phrase ~75 times, in different ways and with
    two microphones. The training notebook mixes these in with its synthetic
    voices, so the model learns how *you* actually say it — the biggest
    single fix for missed wake-ups."""
    import sounddevice as sd
    from openwakeword.vad import VAD
    vad = VAD()
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(VOICE_DIR.glob("*.wav")))
    total = sum(k for _, rounds in VOICE_PARTS for _, k in rounds)
    print(f"Recording \"{WAKE_PHRASE}\" about {total} times. Say it whenever you see ▶, "
          f"with a short pause after each one — no need to wait for a cue.")
    if n:
        print(f"({n} clips already in {VOICE_DIR}; new ones are added. Delete the folder to start over.)")
    try:
        for p, (mic, rounds) in enumerate(VOICE_PARTS, 1):
            print(f"\n=== Part {p}/{len(VOICE_PARTS)}: {mic} ===")
            if p > 1:
                input("  Switch microphones now, then press Enter...")
            device = _pick_mic()
            for r, (how, count) in enumerate(rounds, 1):
                input(f"\nRound {r}/{len(rounds)}: say it {how}, {count} times. Press Enter to start...")
                vad.reset_states()
                clips = _record_round(count, device, vad)
                first = n
                for clip in clips:
                    _save_wav(VOICE_DIR / f"voice_{n:04d}.wav", clip)
                    n += 1
                if input("  Saved. Press Enter for the next round, or P to play this round back: ").strip().lower() == "p":
                    for clip in clips:
                        sd.play(clip.astype(np.float32) / 32768, SAMPLE_RATE)
                        sd.wait()
                        time.sleep(0.25)
                    if input("  Keep this round? (Enter = yes, N = record it again): ").strip().lower() == "n":
                        for i in range(first, n):
                            (VOICE_DIR / f"voice_{i:04d}.wav").unlink(missing_ok=True)
                        n = first
                        print("  Removed — run myvoice again at the end to top up.")
    except KeyboardInterrupt:
        print("\nStopped early — the clips so far are kept.")
    zip_path = wakeword.WAKE_DIR / "my_voice.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for wav in sorted(VOICE_DIR.glob("*.wav")):
            z.write(wav, wav.name)
    print(f"\n{len(list(VOICE_DIR.glob('*.wav')))} clips → {zip_path}\n"
          f"Listen to a few in {VOICE_DIR} first; then give the zip to the training notebook (cell 5).")


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
