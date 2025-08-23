# chord_audi.py — streamlit-friendly, switchable synth voices (no FluidSynth)

from __future__ import annotations

import io
import re
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf

# ---------------- Globals ----------------
SAMPLE_RATE   = 22050
DEFAULT_OCT   = 4
DEFAULT_TEMPO = 120

# Metronome (downbeat louder)
MET_CLICK_HZ     = 2000.0
MET_CLICK_MS     = 12.0
MET_CLICK_GAIN_1 = 0.42
MET_CLICK_GAIN   = 0.28

# Beat dynamics (re-strike chord every beat in 4/4)
BEAT_VELO_1 = 0.95
BEAT_VELO   = 0.75

# ---------------- Chord parsing ----------------
_SEMITONE = {"C":0,"Cs":1,"Db":1,"D":2,"Ds":3,"Eb":3,"E":4,
             "F":5,"Fs":6,"Gb":6,"G":7,"Gs":8,"Ab":8,
             "A":9,"As":10,"Bb":10,"B":11}

_RE_MIN = re.compile(r"(?i)\bmin(?=\d|$)")
_RE_MAJ = re.compile(r"(?i)\bmaj(?=\d|$)")

def _normalize_rest(rest: str) -> str:
    s = rest.strip()
    if not s: return s
    s = _RE_MIN.sub("m", s)
    s = _RE_MAJ.sub("maj", s)
    s = s.replace("majs9","maj9").replace("maj911s","maj9#11").replace("majs911s","maj9#11")
    s = s.replace("no3d","")
    s = re.sub(r"(?<=\d)s","#",s)  # 11s -> 11#
    return s

def _letter_acc_to_semi(letter: str, acc: str) -> int:
    base = letter.upper()
    if base not in _SEMITONE:
        raise ValueError(f"Unknown pitch letter '{letter}'")
    offs = sum(1 if ch=="s" else -1 for ch in acc)
    return (_SEMITONE[base] + offs) % 12

def _parse_root_bass(token: str) -> Tuple[int, Optional[int], str]:
    token = token.strip()

    # Exact slash like F/A, D/Fs, Bb/C
    m_plain = re.match(r"^([A-G])([sb]*)/([A-G])([sb]*)$", token)
    if m_plain:
        rL,rA,bL,bA = m_plain.groups()
        return _letter_acc_to_semi(rL, rA or ""), _letter_acc_to_semi(bL, bA or ""), ""

    # General split
    if "/" in token:
        main, bass = token.split("/", 1)
        main, bass = main.strip(), bass.strip()
    else:
        main, bass = token, None

    m = re.match(r"^([A-G])([sb]*)?(.*)$", main)
    if not m:
        raise ValueError(f"Cannot parse '{token}'")
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
            bass_semi = _letter_acc_to_semi(bL, bA or "")

    return root, bass_semi, _normalize_rest(rest)

def _base_quality_intervals(rest:str)->List[int]:
    r=rest.lower()
    if "sus2" in r: return [0,2,7]
    if "sus4" in r: return [0,5,7]
    if "dim"  in r: return [0,3,6]
    if "aug"  in r: return [0,4,8]
    if re.search(r"(^|[^a-z])m(?!aj)",r): return [0,3,7]  # minor (not 'maj')
    return [0,4,7]  # major triad default

def _apply_extensions(base:List[int],rest:str)->List[int]:
    r=rest.lower(); s=set(base)
    # 6 / 7 / maj7
    if "maj7" in r:
        s.add(11)
    elif re.search(r"(?<!\d)7(?!\d)", r):
        s.add(10)
    if re.search(r"(?<!\d)6(?!\d)", r):
        s.add(9)
    # add & compound tensions
    if "add9" in r or re.search(r"(?<!\d)9(?!\d)", r):  s.add(14)
    if "add11" in r or re.search(r"(?<!\d)11(?!\d)", r): s.add(17)
    if "add13" in r or re.search(r"(?<!\d)13(?!\d)", r): s.add(21)
    # alterations
    if "b9"  in r: s.add(13)
    if "#9"  in r: s.add(15)
    if "#11" in r: s.add(18)
    if "b13" in r: s.add(20)
    return sorted(s)

def chord_token_to_midinums(token:str, base_oct:int=DEFAULT_OCT)->List[int]:
    token = token.strip()
    if not token:
        raise ValueError("Empty chord token")
    if token.endswith("/"):
        raise ValueError(f"Incomplete slash chord '{token}'")

    root, bass, rest = _parse_root_bass(token)
    ints = _apply_extensions(_base_quality_intervals(rest), rest)

    root_midi = root + 12 * (base_oct + 1)     # chord around middle register
    notes = [root_midi + i for i in ints]
    if bass is not None:
        notes.append(bass + 12 * base_oct)     # slash bass one octave below

    out = sorted(set(int(n) for n in notes if 0 <= n <= 127))
    if not out:
        raise ValueError(f"No playable pitches for '{token}'")
    return out

# ---------------- Synth building blocks ----------------
def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    """
    a,d,r in seconds; s is sustain level in [0,1].
    Always returns a float32 envelope of length n without shape/broadcast issues.
    """
    A = max(int(round(a * sr)), 0)
    D = max(int(round(d * sr)), 0)
    R = max(int(round(r * sr)), 0)

    # Fit segments into n samples: allocate sustain as the remainder
    S = n - (A + D + R)
    if S < 0:
        # Not enough room: first try to shrink R (release), then D, then A
        overflow = -(S)
        take = min(R, overflow); R -= take; overflow -= take
        if overflow > 0:
            take = min(D, overflow); D -= take; overflow -= take
        if overflow > 0:
            take = min(A, overflow); A -= take; overflow -= take
        S = 0
        # Final sanity: clip negatives
        A = max(A, 0); D = max(D, 0); R = max(R, 0)

    env = np.zeros(n, dtype=np.float32)

    # Attack
    if A > 0:
        env[:A] = np.linspace(0.0, 1.0, A, endpoint=False, dtype=np.float32)
    else:
        env[:0] = np.array([], dtype=np.float32)

    # Decay
    if D > 0:
        env[A:A + D] = np.linspace(1.0, float(s), D, endpoint=False, dtype=np.float32)

    # Sustain (flat)
    if S > 0:
        env[A + D:A + D + S] = float(s)

    # Release fills the remaining tail exactly
    tail = n - (A + D + S)
    if tail > 0:
        if R > 0:
            env[A + D + S:] = np.linspace(float(s), 0.0, tail, endpoint=True, dtype=np.float32)
        else:
            # No release time: hold sustain (or silence if s=0)
            env[A + D + S:] = float(s)

    return env


def _midi_to_hz(m:int)->float:
    return 440.0 * (2.0 ** ((m-69)/12.0))

def _click(sr:int, ms:float, hz:float, gain:float)->np.ndarray:
    n = int(sr * ms / 1000.0)
    t = np.arange(n)/sr
    env = np.exp(-12.0*t).astype(np.float32)
    return (gain*np.sin(2*np.pi*hz*t)*env).astype(np.float32)

def _overlay(buf: np.ndarray, seg: np.ndarray, start: int) -> None:
    end = min(start + len(seg), len(buf))
    if end > start:
        buf[start:end] += seg[:end-start]

# ----------- Instrument voices (choose via name) -----------
def _voice_piano(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, dtype=np.float32) / sr
    w = (1.00*np.sin(2*np.pi*freq*t)
       + 0.35*np.sin(2*np.pi*(2*freq)*t)
       + 0.20*np.sin(2*np.pi*(3*freq)*t))
    env = _adsr(n, sr, a=0.004, d=0.06, s=0.5, r=0.12)
    return (amp*w*env).astype(np.float32)

def _voice_epiano(freq, dur, sr, amp):
    n = int(sr*dur)
    t = np.arange(n, dtype=np.float32) / sr   # ✅ fixed
    base = np.sign(np.sin(2*np.pi*freq*t))  # square-ish
    w = 0.8*base + 0.2*np.sin(2*np.pi*2*freq*t)
    env = _adsr(n, sr, a=0.002, d=0.10, s=0.65, r=0.18)
    trem = 0.95 + 0.05*np.sin(2*np.pi*5*t)
    return (amp*w*env*trem).astype(np.float32)

def _voice_organ(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, dtype=np.float32) / sr 
    # drawbars like: 8', 4', 2 2/3', 2'
    w = (0.9*np.sin(2*np.pi*freq*t)
       + 0.5*np.sin(2*np.pi*(2*freq)*t)
       + 0.35*np.sin(2*np.pi*(3*freq/2)*t))
    env = _adsr(n, sr, a=0.005, d=0.02, s=0.95, r=0.01)
    return (amp*w*env).astype(np.float32)

def _saw(t, f):  # helper
    return 2*(t*f - np.floor(0.5 + t*f))

def _voice_saw(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, np.float32)/sr
    w = _saw(t, freq) + 0.4*_saw(t, freq*0.997)
    env = _adsr(n, sr, a=0.004, d=0.06, s=0.7, r=0.15)
    return (amp*w*env).astype(np.float32)

def _voice_square(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, np.float32)/sr
    w = np.sign(np.sin(2*np.pi*freq*t))
    env = _adsr(n, sr, a=0.004, d=0.05, s=0.7, r=0.12)
    return (amp*w*env).astype(np.float32)

def _voice_guitar(freq, dur, sr, amp):
    # plucky with exponential decay + a little inharmonic overtone
    n = int(sr*dur); t = np.arange(n, dtype=np.float32) / sr
    w = (np.sin(2*np.pi*freq*t)
       + 0.25*np.sin(2*np.pi*(2.01*freq)*t)
       + 0.12*np.sin(2*np.pi*(3.02*freq)*t))
    env = _adsr(n, sr, a=0.0015, d=0.06, s=0.4, r=0.20)
    return (amp*w*env).astype(np.float32)

def _voice_marimba(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, np.float32)/sr
    w = (np.sin(2*np.pi*freq*t)
       + 0.6*np.sin(2*np.pi*(3.9*freq/2)*t))  # inharmonic partial
    env = _adsr(n, sr, a=0.001, d=0.10, s=0.0, r=0.25)
    return (amp*w*env).astype(np.float32)

def _voice_pad(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, np.float32)/sr
    w = (np.sin(2*np.pi*freq*t)
       + 0.5*np.sin(2*np.pi*(2*freq)*t)
       + 0.4*np.sin(2*np.pi*(1.5*freq)*t))
    env = _adsr(n, sr, a=0.20, d=0.40, s=0.9, r=0.80)
    chorus = 0.5*np.sin(2*np.pi*(freq*0.997)*t)
    return (amp*(w+chorus)*env).astype(np.float32)

def _voice_sine(freq, dur, sr, amp):
    n = int(sr*dur); t = np.arange(n, np.float32)/sr
    w = np.sin(2*np.pi*freq*t)
    env = _adsr(n, sr, a=0.01, d=0.08, s=0.8, r=0.12)
    return (amp*w*env).astype(np.float32)

_VOICES = {
    "piano":   _voice_piano,
    "epiano":  _voice_epiano,
    "organ":   _voice_organ,
    "guitar":  _voice_guitar,
    "square":  _voice_square,
    "saw":     _voice_saw,
    "sine":    _voice_sine,
    "pad":     _voice_pad,
    "marimba": _voice_marimba,
}

# ---------------- Public API ----------------
def generate_audio(
    chords: List[str],
    chord_dur: float = 2.0,          # seconds per chord (1 bar @120 BPM)
    tempo: int = DEFAULT_TEMPO,      # BPM (metronome & beat spacing)
    instrument: str = "piano",       # <— choose voice here
    octave: int = DEFAULT_OCT,
    include_metronome: bool = False,
    four_four_restrike: bool = True  # re-strike on beats 1 2 3 4
) -> Tuple[bytes, List[str], List[str]]:
    """
    Returns (wav_bytes, skipped, errors).

    • instrument ∈ {"piano","epiano","organ","guitar","square","saw","sine","pad","marimba"}
    • Parses tokenizer chords (slash chords like F/A, D/Fs; sus/dim/7/9/11/13/add…)
    • 4/4 feel: hits on each beat (downbeat louder) if four_four_restrike=True
    """
    if instrument not in _VOICES:
        raise ValueError(f"Unknown instrument '{instrument}'. Choose from: {', '.join(_VOICES)}")

    if not chords:
        raise ValueError("No chords provided.")

    voice = _VOICES[instrument]
    skipped: List[str] = []
    errors:  List[str] = []

    spb = 60.0 / float(tempo)   # seconds per beat
    beats_per_bar = 4

    total_seconds = chord_dur * len(chords)
    audio = np.zeros(int(SAMPLE_RATE * total_seconds) + 1, dtype=np.float32)

    click_down = _click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_HZ, MET_CLICK_GAIN_1)
    click      = _click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_HZ, MET_CLICK_GAIN)

    t0 = 0.0
    for ch in chords:
        ch = ch.strip()
        if not ch:
            skipped.append(ch)
            t0 += chord_dur
            continue

        try:
            mids = chord_token_to_midinums(ch, base_oct=octave)
        except Exception as e:
            skipped.append(ch)
            errors.append(f"{ch} ➜ {e}")
            t0 += chord_dur
            continue

        if four_four_restrike:
            n_beats = max(1, int(np.floor(chord_dur / spb + 1e-6)))
            for b in range(n_beats):
                beat_time = t0 + b * spb
                velo = BEAT_VELO_1 if (b % beats_per_bar == 0) else BEAT_VELO
                dur  = min(0.45 * spb, chord_dur - b*spb)
                if dur <= 0: break
                start_idx = int(SAMPLE_RATE * beat_time)
                for m in mids:
                    note = voice(_midi_to_hz(m), dur, SAMPLE_RATE, amp=velo / max(len(mids), 3))
                    _overlay(audio, note, start_idx)
                if include_metronome:
                    _overlay(audio, click_down if (b % beats_per_bar == 0) else click, start_idx)
        else:
            # sustain once for the whole bar
            start_idx = int(SAMPLE_RATE * t0)
            for m in mids:
                note = voice(_midi_to_hz(m), chord_dur, SAMPLE_RATE, amp=0.9 / max(len(mids), 3))
                _overlay(audio, note, start_idx)
            if include_metronome:
                n_beats = max(1, int(np.floor(chord_dur / spb + 1e-6)))
                for b in range(n_beats):
                    idx = start_idx + int(b * spb * SAMPLE_RATE)
                    _overlay(audio, click_down if (b % beats_per_bar == 0) else click, idx)

        t0 += chord_dur

    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue(), skipped, errors


# ---------------- Utility ----------------
def test_tone(freq: float = 440.0, duration: float = 1.0) -> bytes:
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), False)
    snd = 0.5 * np.sin(2*np.pi*freq*t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, snd, SAMPLE_RATE, format="WAV")
    return buf.getvalue()
