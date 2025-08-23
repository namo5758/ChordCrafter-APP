# chord_audi.py
from __future__ import annotations

import io
import re
import inspect
import zipfile
import tempfile
import requests
from typing import List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import soundfile as sf
import pretty_midi


# --------------------------- Globals & Defaults ------------------------------

SAMPLE_RATE   = 22050
DEFAULT_OCT   = 4
DEFAULT_TEMPO = 120          # BPM
DEFAULT_PROG  = 0            # 0 = Acoustic Grand Piano (GM)

# Metronome (click on each beat; beat 1 accent)
MET_CLICK_FREQ     = 2000.0
MET_CLICK_MS       = 12.0
MET_CLICK_GAIN_1   = 0.42    # downbeat
MET_CLICK_GAIN_OTH = 0.28    # beats 2-4

# Where to look for a local SoundFont inside the repo
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SF_CANDIDATES = [
    _THIS_DIR / "assets"  / "sf2" / "GXSCC_gm_033.sf2",  # correct folder
    _THIS_DIR / "assests" / "sf2" / "GXSCC_gm_033.sf2",  # misspelled folder
]

# Cloud-friendly auto-download target (use /tmp on Streamlit Cloud)
SF2_URL      = "https://www.woolyss.com/chipmusic/chipmusic-soundfonts/GXSCC_gm_033.zip"
SF2_BASENAME = "GXSCC_gm_033.sf2"
SF2_DIR      = Path(tempfile.gettempdir()) / "sf2"
SF2_PATH     = SF2_DIR / SF2_BASENAME


# --------------------------- Token → MIDI Parser -----------------------------

# letter to semitone (plus common enharmonics)
_SEMITONE = {
    "C": 0,  "Cs": 1,  "Db": 1,
    "D": 2,  "Ds": 3,  "Eb": 3,
    "E": 4,
    "F": 5,  "Fs": 6,  "Gb": 6,
    "G": 7,  "Gs": 8,  "Ab": 8,
    "A": 9,  "As": 10, "Bb": 10,
    "B": 11,
}

# normalize odd spellings from the tokenizer
_RE_MIN = re.compile(r"(?i)\bmin(?=\d|$)")
_RE_MAJ = re.compile(r"(?i)\bmaj(?=\d|$)")

def _normalize_rest(rest: str) -> str:
    s = rest.strip()
    if not s:
        return s
    s = _RE_MIN.sub("m", s)
    s = s.replace("majs9", "maj9")
    s = s.replace("maj911s", "maj9#11").replace("majs911s", "maj9#11")
    s = s.replace("no3d", "")
    s = re.sub(r"(?<=\d)s", "#", s)  # 11s -> 11#
    return s

def _letter_acc_to_semi(letter: str, acc: str) -> int:
    base = letter.upper()
    if base not in _SEMITONE:
        raise ValueError(f"Unknown pitch letter '{letter}'")
    base_semi = _SEMITONE[base]
    offs = 0
    for ch in acc:
        if ch == 's': offs += 1
        elif ch == 'b': offs -= 1
    return (base_semi + offs) % 12

def _parse_root_bass(token: str) -> Tuple[int, Optional[int], str]:
    """
    Split token into (root_semi, optional bass_semi, rest).
    Handles:
      - F/A, D/Fs, Bb/C
      - Gsus4 (does NOT become G# + 'us4')
    """
    token = token.strip()

    # Exact slash chord like F/A or D/Fs
    m_plain_slash = re.match(r"^([A-G])([sb]*)/([A-G])([sb]*)$", token)
    if m_plain_slash:
        rL, rA, bL, bA = m_plain_slash.groups()
        root_semi = _letter_acc_to_semi(rL, rA or "")
        bass_semi = _letter_acc_to_semi(bL, bA or "")
        return root_semi, bass_semi, ""

    # General case
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

    root_semi = _letter_acc_to_semi(letter, acc)

    bass_semi: Optional[int] = None
    if bass:
        m2 = re.match(r"^([A-G])([sb]*)", bass)  # stop at non-[sb]
        if m2:
            bL, bA = m2.groups()
            bass_semi = _letter_acc_to_semi(bL, (bA or ""))

    return root_semi, bass_semi, _normalize_rest(rest)

def _base_quality_intervals(rest: str) -> List[int]:
    r = rest.lower()
    if "sus2" in r: return [0, 2, 7]
    if "sus4" in r: return [0, 5, 7]
    if "dim"  in r: return [0, 3, 6]
    if "aug"  in r: return [0, 4, 8]
    if re.search(r"(^|[^a-z])m(?!aj)", r):  # minor (but not 'maj')
        return [0, 3, 7]
    return [0, 4, 7]  # default major

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

    # compound tensions
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
    Convert token into MIDI notes.
    - If no quality given (e.g., 'F/A'), assume MAJOR triad on root.
    - Root chord voiced around (base_octave+1); slash bass one octave below.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty chord token")
    if token.endswith("/"):
        raise ValueError(f"Incomplete slash chord '{token}'")

    # normalize rare prefix like 'sA...' (seen in some dumps)
    if re.match(r"^s([A-G])", token):
        token = token[1] + "s" + token[1:]

    root_semi, bass_semi, rest = _parse_root_bass(token)
    base = _base_quality_intervals(rest)
    ints = _apply_extensions(base, rest)

    root_midi = root_semi + 12 * (base_octave + 1)
    notes = [root_midi + i for i in ints]

    if bass_semi is not None:
        notes.append(bass_semi + 12 * base_octave)

    out = sorted(set(int(n) for n in notes if 0 <= n <= 127))
    if not out:
        raise ValueError(f"No playable pitches for '{token}'")
    return out


# --------------------------- Rendering Helpers -------------------------------

def _sine_click(sr: int, ms: float, hz: float, gain: float) -> np.ndarray:
    n = int(sr * ms / 1000.0)
    t = np.arange(n) / sr
    env = np.exp(-12.0 * t)
    return gain * np.sin(2 * np.pi * hz * t) * env

def _overlay(audio: np.ndarray, click: np.ndarray, start_idx: int) -> None:
    end = min(start_idx + len(click), len(audio))
    seg = end - start_idx
    if seg > 0:
        audio[start_idx:end] += click[:seg]

def _try_existing_path(p: Union[str, Path]) -> Optional[str]:
    if not p:
        return None
    p = str(p)
    if p.lower().endswith(".sf2") and Path(p).exists():
        return p
    return None

def _download_sf2_to_tmp() -> str:
    SF2_DIR.mkdir(parents=True, exist_ok=True)
    if SF2_PATH.exists():
        return str(SF2_PATH)

    # download and extract the first .sf2 inside the ZIP
    r = requests.get(SF2_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        member = next((m for m in zf.namelist() if m.lower().endswith(".sf2")), None)
        if not member:
            raise RuntimeError("No .sf2 found inside downloaded ZIP")
        with zf.open(member) as fsrc, open(SF2_PATH, "wb") as fdst:
            fdst.write(fsrc.read())
    if not SF2_PATH.exists():
        raise RuntimeError(f"Failed to materialize SoundFont at {SF2_PATH}")
    return str(SF2_PATH)

def _resolve_soundfont_path(user_path: Optional[Union[str, Path]]) -> str:
    """
    Priority:
      1) user_path (if provided and exists)
      2) local candidates (./assets/sf2/... or ./assests/sf2/...)
      3) auto-download to /tmp and use that
    """
    if user_path:
        found = _try_existing_path(user_path)
        if found:
            return found

    for cand in _DEFAULT_SF_CANDIDATES:
        found = _try_existing_path(cand)
        if found:
            return found

    return _download_sf2_to_tmp()

def _render_with_fluidsynth(pm: pretty_midi.PrettyMIDI, sf2_path: str, sr: int) -> np.ndarray:
    """
    Render PrettyMIDI to audio using a SoundFont via FluidSynth.

    pretty_midi signatures differ by version:
      - fluidsynth(sample_rate=..., sf2_path=...)
      - fluidsynth(samplerate=...,  sf2_path=...)
      - fluidsynth(fs=...,          sf2_path=...)
    Detect parameter names and call with keywords to avoid positional mixups.
    """
    sig = inspect.signature(pm.fluidsynth)
    params = sig.parameters

    kwargs = {}
    if "sample_rate" in params:
        kwargs["sample_rate"] = sr
    elif "samplerate" in params:
        kwargs["samplerate"] = sr
    elif "fs" in params:
        kwargs["fs"] = sr

    if "sf2_path" in params:
        kwargs["sf2_path"] = sf2_path
    else:
        kwargs["sf2_path"] = sf2_path  # defensive fallback

    audio = pm.fluidsynth(**kwargs)
    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)


# --------------------------- Public API --------------------------------------

def generate_audio(
    chords: List[str],
    chord_dur: float = 1.0,           # seconds per chord (treat as 1 bar of 4/4)
    tempo: int = DEFAULT_TEMPO,       # metronome tempo
    soundfont_path: Optional[Union[str, Path]] = None,  # None → resolve/download
    program: int = DEFAULT_PROG,      # GM program (0..127)
    octave: int = DEFAULT_OCT,
    include_metronome: bool = False,
    placement: str = "on_1",          # "on_1" | "on_all" | "stabs_234"
    stab_ms: float = 110.0,           # length of short stabs (ms) for "stabs_234"
) -> Tuple[bytes, List[str], List[str]]:
    """
    Returns (wav_bytes, skipped, errors)

    placement styles (4/4):
      - "on_1":      hold the chord for the whole bar (hit on beat 1)
      - "on_all":    hit on each beat (1,2,3,4), quarter-note staccato
      - "stabs_234": rest on 1; short stabs on 2,3,4
    """
    if not chords:
        raise ValueError("No chords provided.")

    sf2 = _resolve_soundfont_path(soundfont_path)

    skipped: List[str] = []
    errors:  List[str] = []

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program))

    # Assume chord_dur == one bar in 4/4
    beats_per_bar = 4
    beat_dur = chord_dur / beats_per_bar
    time_cursor = 0.0

    for raw in chords:
        token = raw.strip()
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

        # schedule notes according to placement
        if placement == "on_1":
            start = time_cursor
            end   = time_cursor + chord_dur
            for m in mids:
                inst.notes.append(pretty_midi.Note(velocity=96, pitch=int(m), start=start, end=end))

        elif placement == "on_all":
            # play on 1,2,3,4 as short quarters
            for b in range(beats_per_bar):
                start = time_cursor + b * beat_dur
                end   = start + max(0.06, 0.9 * beat_dur)  # slightly detached
                for m in mids:
                    inst.notes.append(pretty_midi.Note(velocity=94, pitch=int(m), start=start, end=end))

        elif placement == "stabs_234":
            # rest on 1; short stabs on 2,3,4
            stab_sec = max(0.04, stab_ms / 1000.0)
            for b in (1, 2, 3):  # beats 2,3,4
                start = time_cursor + b * beat_dur
                end   = min(start + stab_sec, time_cursor + chord_dur)
                for m in mids:
                    inst.notes.append(pretty_midi.Note(velocity=98, pitch=int(m), start=start, end=end))
        else:
            # fallback: same as on_1
            start = time_cursor
            end   = time_cursor + chord_dur
            for m in mids:
                inst.notes.append(pretty_midi.Note(velocity=96, pitch=int(m), start=start, end=end))

        time_cursor += chord_dur

    pm.instruments.append(inst)

    # Render audio
    audio = _render_with_fluidsynth(pm, sf2, SAMPLE_RATE)

    # Optional metronome (quarter notes with accent on beat 1)
    if include_metronome:
        click1 = _sine_click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_FREQ, MET_CLICK_GAIN_1)
        clicko = _sine_click(SAMPLE_RATE, MET_CLICK_MS, MET_CLICK_FREQ, MET_CLICK_GAIN_OTH)

        sec_per_beat = 60.0 / float(tempo)
        total_time   = pm.get_end_time()
        n_beats = int(np.floor(total_time / sec_per_beat)) + 1
        for b in range(n_beats):
            idx = int(b * sec_per_beat * SAMPLE_RATE)
            _overlay(audio, click1 if (b % beats_per_bar == 0) else clicko, idx)

        audio = np.clip(audio, -1.0, 1.0)

    # WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue(), skipped, errors


# --------------------------- Utilities ---------------------------------------

def test_tone(freq: float = 440.0, duration: float = 1.0) -> bytes:
    """
    Quick A/B check: returns a plain sine wave tone as WAV bytes.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    snd = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, snd, SAMPLE_RATE, format="WAV")
    return buf.getvalue()
