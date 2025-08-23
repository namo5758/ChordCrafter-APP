# chord_audi.py  —  self-contained, Streamlit-friendly (no FluidSynth needed)

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
DEFAULT_PROG  = 0  # kept for API compatibility (ignored by synth)

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

# ---------------- Lightweight piano-ish synth ----------------
def _adsr_envelope(n_samples:int, sr:int, a=0.005, d=0.08, s=0.6, r=0.15)->np.ndarray:
    """Simple ADSR in seconds → samples; A/D/R are times, S is sustain level."""
    A = int(sr * a); D = int(sr * d); R = int(sr * r)
    S = max(0, n_samples - (A + D + R))

    env = np.zeros(n_samples, dtype=np.float32)
    # Attack
    if A > 0:
        env[:A] = np.linspace(0.0, 1.0, A, endpoint=False)
    # Decay
    if D > 0:
        start = A
        end   = A + D
        env[start:end] = np.linspace(1.0, s, D, endpoint=False)
    # Sustain
    if S > 0:
        start = A + D
        end   = A + D + S
        env[start:end] = s
    # Release
    if R > 0:
        start = A + D + S
        end   = n_samples
        env[start:end] = np.linspace(s, 0.0, R, endpoint=True)
    return env

def _note_wave(freq: float, dur_s: float, sr:int, amp:float) -> np.ndarray:
    """A gentle, piano-ish tone: 3 harmonics (1,2,3) with ADSR."""
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float32) / sr
    # harmonics
    w = (1.00*np.sin(2*np.pi*freq*t)
       + 0.35*np.sin(2*np.pi*(2*freq)*t)
       + 0.20*np.sin(2*np.pi*(3*freq)*t))
    env = _adsr_envelope(n, sr, a=0.004, d=0.06, s=0.5, r=0.12)
    return (amp * w * env).astype(np.float32)

def _midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def _sine_click(sr:int, ms:float, hz:float, gain:float)->np.ndarray:
    n = int(sr * ms / 1000.0)
    t = np.arange(n) / sr
    env = np.exp(-12.0 * t).astype(np.float32)
    return (gain * np.sin(2*np.pi*hz*t) * env).astype(np.float32)

def _overlay(buf: np.ndarray, seg: np.ndarray, start: int) -> None:
    end = min(start + len(seg), len(buf))
    if end > start:
        buf[start:end] += seg[:end-start]

# ---------------- Public API (no FluidSynth) ----------------
def generate_audio(
    chords: List[str],
    chord_dur: float = 1.0,          # seconds per chord (1 bar if 4/4 & tempo matches)
    tempo: int = DEFAULT_TEMPO,      # BPM (metronome & beat spacing)
    soundfont_path: Optional[str] = None,  # kept for API compatibility; ignored
    program: int = DEFAULT_PROG,     # kept for API compatibility; ignored
    octave: int = DEFAULT_OCT,
    include_metronome: bool = False,
    four_four_restrike: bool = True  # re-strike chord on each beat: 1 2 3 4
) -> Tuple[bytes, List[str], List[str]]:
    """
    Returns (wav_bytes, skipped, errors).

    • Parses your tokenizer chords (supports slash chords like F/A, D/Fs, …).
    • Renders with a built-in synth (no FluidSynth or .sf2 needed).
    • 4/4 pattern: strikes the chord every beat (downbeat louder) if four_four_restrike=True.
    """
    if not chords:
        raise ValueError("No chords provided.")

    skipped: List[str] = []
    errors:  List[str] = []

    # Beat timing
    spb = 60.0 / float(tempo)  # seconds per beat
    beats_per_bar = 4
    bar_seconds = beats_per_bar * spb

    # If the user passes chord_dur as "per chord", we keep it.
    # For 4/4 feel, you probably want chord_dur == bar_seconds.
    # We'll still re-strike per beat inside that duration if asked.

    total_seconds = chord_dur * len(chords)
    audio = np.zeros(int(SAMPLE_RATE * total_seconds) + 1, dtype=np.float32)

    # Precompute metronome clicks if needed
    click_down = _sine_click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_HZ, MET_CLICK_GAIN_1)
    click      = _sine_click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_HZ, MET_CLICK_GAIN)

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

        # How many beats inside this chord segment?
        if four_four_restrike:
            n_beats = max(1, int(np.floor(chord_dur / spb + 1e-6)))
            for b in range(n_beats):
                beat_time = t0 + b * spb
                # Downbeat louder
                velo = BEAT_VELO_1 if (b % beats_per_bar == 0) else BEAT_VELO
                # Shorter strike per beat
                dur = min(0.45 * spb, chord_dur - b*spb)  # a short, plucky envelope
                if dur <= 0: break
                seg_n = int(SAMPLE_RATE * (beat_time))
                for m in mids:
                    freq = _midi_to_hz(m)
                    note = _note_wave(freq, dur, SAMPLE_RATE, amp=velo / max(len(mids), 3))
                    _overlay(audio, note, seg_n)

                # Metronome on top
                if include_metronome:
                    _overlay(audio, click_down if (b % beats_per_bar == 0) else click, seg_n)
        else:
            # Sustain the whole chord_dur, single strike
            seg_n = int(SAMPLE_RATE * t0)
            dur = chord_dur
            for m in mids:
                freq = _midi_to_hz(m)
                note = _note_wave(freq, dur, SAMPLE_RATE, amp=0.9 / max(len(mids), 3))
                _overlay(audio, note, seg_n)

            if include_metronome:
                # place clicks each beat across this bar
                n_beats = max(1, int(np.floor(chord_dur / spb + 1e-6)))
                for b in range(n_beats):
                    seg_b = seg_n + int(b * spb * SAMPLE_RATE)
                    _overlay(audio, click_down if (b % beats_per_bar == 0) else click, seg_b)

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
