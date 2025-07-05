import numpy as np
import io, soundfile as sf
import re
from typing import List, Tuple
from pychord import Chord

SAMPLE_RATE = 22_050    
AMP         = 0.30      
ENV_DECAY   = 3.0        
DEFAULT_OCT = 4          

_RE_MIN = re.compile(r"min(?=\d|$)", re.IGNORECASE)
_RE_MAJ = re.compile(r"maj(?=\d|$)", re.IGNORECASE)

def normalize_chord_string(name: str) -> str:
    if "/" in name:
        name = name.split("/", 1)[0]

    name = name.strip()
    name = re.sub(r"([A-Ga-g])s", r"\1#", name)
    name = name.replace("♯", "#").replace("♭", "b")
    name = _RE_MIN.sub("m", name)      
    name = _RE_MAJ.sub("maj", name)    

    return name

_BASE_SEMI = dict(zip("C D E F G A B".split(), [0,2,4,5,7,9,11]))

def note_str_to_semitone(note: str) -> int:
    """
    'C', 'F#', 'Gb', 'E##', 'Abb' → 0–11
    """
    note = note.strip()
    if not note:
        raise ValueError("empty note")

    base = note[0].upper()
    if base not in _BASE_SEMI:
        raise ValueError(f"unknown base '{base}'")

    offset = 0
    for ch in note[1:]:
        if ch in ("#", "♯"):
            offset += 1
        elif ch in ("b", "♭"):
            offset -= 1
    return (_BASE_SEMI[base] + offset) % 12

def midi_to_hz(midinum: int) -> float:
    return 440.0 * 2 ** ((midinum - 69) / 12)

def chord_to_freqs(raw_name: str, octave: int = DEFAULT_OCT) -> List[float]:
    raw_name = re.sub(r"add13", "13", raw_name, flags=re.IGNORECASE) 
    name     = normalize_chord_string(raw_name)                       
    chord    = Chord(name)                                            

    root_semi = note_str_to_semitone(str(chord.root))                
    root_midi = root_semi + 12 * (octave + 1)                        

    freqs: List[float] = []
    for n in chord.components():                                     
        offset = note_str_to_semitone(str(n)) - root_semi
        freqs.append(midi_to_hz(root_midi + offset))

    return freqs

def synth_sine(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return AMP * np.sin(2 * np.pi * freq * t) * np.exp(-ENV_DECAY * t)

def generate_audio(chords: List[str], chord_dur: float = 1.0) -> Tuple[bytes, List[str], List[str]]:
    total_len = int(SAMPLE_RATE * chord_dur * len(chords))
    audio = np.zeros(total_len, dtype=np.float32)

    skipped: List[str] = []
    errors:  List[str] = []

    for idx, raw in enumerate(chords):
        try:
            freqs = chord_to_freqs(raw)
        except Exception as e:
            skipped.append(raw)
            errors.append(f"{raw} ➜ {e}")
            continue

        start = int(idx * chord_dur * SAMPLE_RATE)
        note_dur  = chord_dur / len(freqs)
        note_samp = int(note_dur * SAMPLE_RATE)

        for j, f in enumerate(freqs):
            s = start + j * note_samp
            e = s + note_samp
            audio[s:e] += synth_sine(f, note_dur)

    audio = np.clip(audio, -1, 1)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue(), skipped, errors

def test_tone(freq: float = 440.0, duration: float = 1.0) -> bytes:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    snd = 0.5 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, snd, SAMPLE_RATE, format="WAV")
    return buf.getvalue()