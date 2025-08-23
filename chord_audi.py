# chord_audi.py
from __future__ import annotations

import io
import re
from typing import List, Optional, Tuple
import numpy as np
import wave

# --------------------------- Globals -----------------------------------------

SAMPLE_RATE   = 22_050
DEFAULT_OCT   = 4
DEFAULT_TEMPO = 120
DEFAULT_VOICE = "epiano"   # "epiano" or "guitar"

# Metronome (simple sine click)
MET_CLICK_FREQ = 2000.0
MET_CLICK_MS   = 12.0
MET_CLICK_GAIN = 0.30

# --------------------------- Token → MIDI Parser -----------------------------

_SEMITONE = {
    "C": 0,  "Cs": 1,  "Db": 1,
    "D": 2,  "Ds": 3,  "Eb": 3,
    "E": 4,
    "F": 5,  "Fs": 6,  "Gb": 6,
    "G": 7,  "Gs": 8,  "Ab": 8,
    "A": 9,  "As": 10, "Bb": 10,
    "B": 11,
}

_RE_MIN = re.compile(r"(?i)\bmin(?=\d|$)")
_RE_MAJ = re.compile(r"(?i)\bmaj(?=\d|$)")

def _normalize_rest(rest: str) -> str:
    s = rest.strip()
    s = _RE_MIN.sub("m", s)                 # "min7" -> "m7"
    s = _RE_MAJ.sub("maj", s)               # keep "maj7"
    s = s.replace("majs9", "maj9")
    s = s.replace("maj911s", "maj9#11").replace("majs911s", "maj9#11")
    s = s.replace("no3d", "")
    s = re.sub(r"add13", "13", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)s", "#", s)         # "11s" -> "11#"
    return s

def _letter_acc_to_semi(letter: str, acc: str) -> int:
    base = letter.upper()
    if base not in _SEMITONE:
        raise ValueError(f"Unknown pitch letter '{letter}'")
    base_semi = _SEMITONE[base]
    offs = 0
    for ch in acc:
        if ch == 's': offs += 1            # "s" means sharp in your vocab
        elif ch == 'b': offs -= 1
    return (base_semi + offs) % 12

def _parse_root_bass(token: str) -> Tuple[int, Optional[int], str]:
    token = token.strip()

    # Exact slash form like F/A, D/Fs, Bb/C …
    m_plain = re.match(r"^([A-G])([sb]*)/([A-G])([sb]*)$", token)
    if m_plain:
        rL, rA, bL, bA = m_plain.groups()
        root = _letter_acc_to_semi(rL, rA or "")
        bass = _letter_acc_to_semi(bL, bA or "")
        return root, bass, ""

    # General split (might have quality/extensions before slash)
    if "/" in token:
        main, bass = token.split("/", 1)
        bass = bass.strip()
    else:
        main, bass = token, None

    m = re.match(r"^([A-G])([sb]*)?(.*)$", main.strip())
    if not m:
        raise ValueError(f"Cannot parse root in '{token}'")
    letter, acc, rest = m.groups()
    acc  = acc or ""
    rest = rest or ""

    # Disambiguate Gsus* vs G + '#'
    if acc == "s" and rest.startswith("us"):
        rest = "sus" + rest[2:]
        acc = ""

    root = _letter_acc_to_semi(letter, acc)

    bass_semi = None
    if bass:
        m2 = re.match(r"^([A-G])([sb]*)", bass)
        if m2:
            bL, bA = m2.groups()
            bass_semi = _letter_acc_to_semi(bL, (bA or ""))

    return root, bass_semi, _normalize_rest(rest)

def _base_quality_intervals(rest: str) -> List[int]:
    r = rest.lower()
    if "sus2" in r: return [0, 2, 7]
    if "sus4" in r: return [0, 5, 7]
    if "dim"  in r: return [0, 3, 6]
    if "aug"  in r: return [0, 4, 8]
    if re.search(r"(^|[^a-z])m(?!aj)", r):  # minor (not 'maj')
        return [0, 3, 7]
    return [0, 4, 7]                        # major

def _apply_extensions(base_ints: List[int], rest: str) -> List[int]:
    r = rest.lower()
    s = set(base_ints)

    # 6/7/maj7
    if "maj7" in r:
        s.add(11)
    elif re.search(r"(?<!\d)7(?!\d)", r):
        s.add(10)
    if re.search(r"(?<!\d)6(?!\d)", r):
        s.add(9)

    # add*
    if "add9"  in r: s.add(14)
    if "add11" in r: s.add(17)
    if "add13" in r: s.add(21)

    # tensions
    if re.search(r"(?<!\d)9(?!\d)",  r): s.add(14)
    if re.search(r"(?<!\d)11(?!\d)", r): s.add(17)
    if re.search(r"(?<!\d)13(?!\d)", r): s.add(21)

    # alterations
    if "b9"  in r: s.add(13)
    if "#9"  in r: s.add(15)
    if "#11" in r: s.add(18)
    if "b13" in r: s.add(20)

    return sorted(s)

def chord_token_to_midinums(token: str, base_octave: int = DEFAULT_OCT) -> List[int]:
    """
    Convert a model token (e.g., 'F/A', 'Gsus4', 'Dminadd9', 'Cmaj7#11') to MIDI note numbers.
    Root chord voiced around (base_octave+1). Slash bass (if present) one octave below.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty chord token")
    if token.endswith("/"):
        raise ValueError(f"Incomplete slash chord '{token}'")

    # normalize rare "sA..." prefix: "sA..." -> "A..." with sharp applied properly
    if re.match(r"^s([A-G])", token):
        token = token[1] + "s" + token[1:]

    root_semi, bass_semi, rest = _parse_root_bass(token)
    base_ints = _base_quality_intervals(rest)
    ints = _apply_extensions(base_ints, rest)

    root_midi = root_semi + 12 * (base_octave + 1)
    notes = [root_midi + i for i in ints]

    if bass_semi is not None:
        notes.append(bass_semi + 12 * base_octave)

    out = sorted(set(int(n) for n in notes if 0 <= n <= 127))
    if not out:
        raise ValueError(f"No playable pitches for '{token}'")
    return out

# --------------------------- DSP helpers -------------------------------------

def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """Robust ADSR; lengths always sum to n (avoids broadcasting errors)."""
    A = int(round(a * sr))
    D = int(round(d * sr))
    R = int(round(r * sr))
    A = max(A, 0); D = max(D, 0); R = max(R, 0)
    if A + D + R > n:
        # shrink proportionally
        scale = n / float(A + D + R)
        A = int(A * scale); D = int(D * scale); R = int(R * scale)
    S = max(n - (A + D + R), 0)

    env = np.zeros(n, dtype=np.float32)
    i = 0
    if A > 0:
        env[i:i+A] = np.linspace(0.0, 1.0, A, endpoint=False, dtype=np.float32)
        i += A
    if D > 0:
        env[i:i+D] = np.linspace(1.0, s, D, endpoint=False, dtype=np.float32)
        i += D
    if S > 0:
        env[i:i+S] = s
        i += S
    if R > 0:
        env[i:i+R] = np.linspace(s, 0.0, R, endpoint=True, dtype=np.float32)
        i += R
    if i < n:
        env[i:] = 0.0
    return env

def _midi_to_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)

def _sine_click(sr: int, ms: float, hz: float, gain: float) -> np.ndarray:
    n = int(sr * ms / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-12.0 * t).astype(np.float32)
    return (gain * np.sin(2 * np.pi * hz * t) * env).astype(np.float32)

# --------------------------- Voices (procedural) ------------------------------

def _voice_epiano(freq: float, dur: float, sr: int, amp: float = 1.0) -> np.ndarray:
    """Simple e-piano-ish additive + mild FM."""
    n = int(sr * dur)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr

    # Carrier with slight FM
    mod = 2 * np.sin(2*np.pi*2.0*t) + 0.5*np.sin(2*np.pi*5.7*t)
    c1  = np.sin(2*np.pi*freq*t + 0.015*mod)
    c2  = 0.5*np.sin(2*np.pi*2*freq*t + 0.01*mod)
    c3  = 0.25*np.sin(2*np.pi*3*freq*t)

    env = _adsr(n, sr, a=0.002, d=0.10, s=0.65, r=0.18)
    tone = (c1 + c2 + c3) * env
    return (amp * tone).astype(np.float32)

def _voice_guitar(freq: float, dur: float, sr: int, amp: float = 1.0) -> np.ndarray:
    """Light pluck: filtered noise burst + decaying sine."""
    n = int(sr * dur)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr

    # short noise "pick" burst
    pick_len = max(4, int(0.004 * sr))
    pick = np.zeros(n, dtype=np.float32)
    pick[:pick_len] = (np.random.rand(pick_len).astype(np.float32) * 2.0 - 1.0)
    # simple lowpass via cumulative mean (very cheap)
    for _ in range(3):
        pick = np.cumsum(pick, dtype=np.float32)
        pick /= (np.arange(n, dtype=np.float32) + 1.0)

    # body tone
    body = np.sin(2*np.pi*freq*t) * np.exp(-4.0*t)
    env  = _adsr(n, sr, a=0.0015, d=0.08, s=0.50, r=0.20)
    tone = 0.9*body + 0.6*pick
    return (amp * tone * env).astype(np.float32)

_VOICES = {
    "epiano": _voice_epiano,
    "guitar": _voice_guitar,
}

# --------------------------- WAV writer (no soundfile) ------------------------

def _to_wav_bytes(mono: np.ndarray, sr: int) -> bytes:
    """Write mono float32 [-1,+1] to 16-bit PCM WAV bytes using stdlib."""
    x = np.clip(mono, -1.0, 1.0)
    x16 = (x * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())
    return buf.getvalue()

# --------------------------- Public API --------------------------------------

def generate_audio(
    chords: List[str],
    chord_dur: float = 1.0,            # seconds per chord
    tempo: int = DEFAULT_TEMPO,        # for metronome only (if enabled)
    voice: str = DEFAULT_VOICE,        # "epiano" or "guitar"
    octave: int = DEFAULT_OCT,
    include_metronome: bool = False,
    pattern: str = "hold",             # "hold" (sustain) or "beats" (1-2-3-4 stabs)
) -> Tuple[bytes, List[str], List[str]]:
    """
    Returns (wav_bytes, skipped, errors).

    - 'chord_dur' is the total seconds allocated per chord token.
      In 4/4, each chord gets 4 beats, so beat_dur = chord_dur / 4.
    - pattern="hold": sustain the chord for 'chord_dur'.
      pattern="beats": re-trigger the chord on each beat with short stabs.
    """
    if not chords:
        raise ValueError("No chords provided.")
    if voice not in _VOICES:
        raise ValueError(f"Unknown voice '{voice}'. Choose from {list(_VOICES)}")

    voice_fn = _VOICES[voice]
    beat_dur = chord_dur / 4.0
    total_secs = chord_dur * len(chords)
    total_n = int(SAMPLE_RATE * total_secs)
    audio = np.zeros(total_n, dtype=np.float32)

    skipped: List[str] = []
    errors:  List[str] = []

    time_cursor = 0.0
    for raw in chords:
        token = (raw or "").strip()
        if not token:
            skipped.append(raw)
            time_cursor += chord_dur
            continue

        try:
            mids = chord_token_to_midinums(token, base_octave=octave)
        except Exception as e:
            skipped.append(raw)
            errors.append(f"{raw} ➜ {e}")
            time_cursor += chord_dur
            continue

        # velocity scaling to avoid clipping on dense chords
        base_velo = 0.85
        per_note_amp = base_velo / max(len(mids), 3)

        if pattern == "beats":
            # short stabs on beats 1-2-3-4
            stab = 0.92 * beat_dur
            for b in range(4):
                start = time_cursor + b * beat_dur
                s_idx = int(start * SAMPLE_RATE)
                for m in mids:
                    note = voice_fn(_midi_to_hz(m), stab, SAMPLE_RATE, amp=per_note_amp)
                    e_idx = min(s_idx + len(note), len(audio))
                    seg = e_idx - s_idx
                    if seg > 0:
                        audio[s_idx:e_idx] += note[:seg]
        else:
            # sustained
            start = time_cursor
            s_idx = int(start * SAMPLE_RATE)
            for m in mids:
                note = voice_fn(_midi_to_hz(m), chord_dur, SAMPLE_RATE, amp=per_note_amp)
                e_idx = min(s_idx + len(note), len(audio))
                seg = e_idx - s_idx
                if seg > 0:
                    audio[s_idx:e_idx] += note[:seg]

        time_cursor += chord_dur

    # Optional metronome (quarter notes through the whole piece)
    if include_metronome:
        click = _sine_click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_FREQ, MET_CLICK_GAIN)
        beats_total = int(np.floor(total_secs / beat_dur + 1e-6))
        for b in range(beats_total + 1):
            idx = int(b * beat_dur * SAMPLE_RATE)
            end = min(idx + len(click), len(audio))
            if end > idx:
                audio[idx:end] += click[:end-idx]

    # final clip and pack
    audio = np.clip(audio, -1.0, 1.0)
    wav_bytes = _to_wav_bytes(audio, SAMPLE_RATE)
    return wav_bytes, skipped, errors

# --------------------------- Quick test --------------------------------------

def test_tone(freq: float = 440.0, duration: float = 1.0) -> bytes:
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    snd = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return _to_wav_bytes(snd, SAMPLE_RATE)
