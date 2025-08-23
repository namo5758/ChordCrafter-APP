# chord_audi.py
from __future__ import annotations

import io, re, inspect
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import pretty_midi

# ---------------- Globals ----------------
SAMPLE_RATE   = 22050
DEFAULT_OCT   = 4
DEFAULT_TEMPO = 120
DEFAULT_PROG  = 0   # 0 = Acoustic Grand Piano

MET_CLICK_FREQ   = 2000.0
MET_CLICK_MS     = 12.0
MET_CLICK_GAIN_1 = 0.42
MET_CLICK_GAIN   = 0.28

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_SF_CANDIDATES = [
    _THIS_DIR / "assets"  / "sf2" / "GXSCC_gm_033.sf2",
    _THIS_DIR / "assests" / "sf2" / "GXSCC_gm_033.sf2",  # common typo
]

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
    s = s.replace("majs9","maj9").replace("maj911s","maj9#11")
    s = re.sub(r"(?<=\d)s","#",s)
    return s

def _letter_acc_to_semi(letter: str, acc: str) -> int:
    base = letter.upper()
    base_semi = _SEMITONE[base]
    offs = sum(1 if ch=="s" else -1 for ch in acc)
    return (base_semi+offs)%12

def _parse_root_bass(token: str) -> Tuple[int, Optional[int], str]:
    token=token.strip()
    m_plain = re.match(r"^([A-G])([sb]*)/([A-G])([sb]*)$",token)
    if m_plain:
        rL,rA,bL,bA=m_plain.groups()
        return _letter_acc_to_semi(rL,rA or ""),_letter_acc_to_semi(bL,bA or ""), ""
    if "/" in token:
        main,bass=token.split("/",1)
        main,bass=main.strip(),bass.strip()
    else:
        main,bass=token,None
    m=re.match(r"^([A-G])([sb]*)?(.*)$",main)
    if not m: raise ValueError(f"Cannot parse '{token}'")
    letter,acc,rest=m.groups(); acc=acc or ""; rest=rest or ""
    if acc=="s" and rest.startswith("us"): rest="sus"+rest[2:]; acc=""
    root=_letter_acc_to_semi(letter,acc)
    bass_semi=None
    if bass:
        m2=re.match(r"^([A-G])([sb]*)",bass)
        if m2:
            bL,bA=m2.groups()
            bass_semi=_letter_acc_to_semi(bL,bA or "")
    return root,bass_semi,_normalize_rest(rest)

def _base_quality_intervals(rest:str)->List[int]:
    r=rest.lower()
    if "sus2" in r: return [0,2,7]
    if "sus4" in r: return [0,5,7]
    if "dim" in r: return [0,3,6]
    if "aug" in r: return [0,4,8]
    if re.search(r"(^|[^a-z])m(?!aj)",r): return [0,3,7]
    return [0,4,7]

def _apply_extensions(base:List[int],rest:str)->List[int]:
    r=rest.lower(); s=set(base)
    if "maj7" in r: s.add(11)
    elif re.search(r"(?<!\d)7(?!\d)",r): s.add(10)
    if re.search(r"(?<!\d)6(?!\d)",r): s.add(9)
    if "add9" in r or re.search(r"(?<!\d)9(?!\d)",r): s.add(14)
    if "add11" in r or re.search(r"(?<!\d)11(?!\d)",r): s.add(17)
    if "add13" in r or re.search(r"(?<!\d)13(?!\d)",r): s.add(21)
    if "b9" in r: s.add(13)
    if "#9" in r: s.add(15)
    if "#11" in r: s.add(18)
    if "b13" in r: s.add(20)
    return sorted(s)

def chord_token_to_midinums(token:str,base_oct:int=DEFAULT_OCT)->List[int]:
    root,bass,rest=_parse_root_bass(token)
    ints=_apply_extensions(_base_quality_intervals(rest),rest)
    root_midi=root+12*(base_oct+1)
    notes=[root_midi+i for i in ints]
    if bass is not None: notes.append(bass+12*base_oct)
    return sorted(set(n for n in notes if 0<=n<=127))

# ---------------- Rendering ----------------
def _sine_click(sr,ms,hz,gain):
    n=int(sr*ms/1000.0)
    t=np.arange(n)/sr; env=np.exp(-12.0*t)
    return gain*np.sin(2*np.pi*hz*t)*env

def _overlay(audio,click,start_idx):
    end=min(start_idx+len(click),len(audio))
    audio[start_idx:end]+=click[:end-start_idx]

def _resolve_soundfont_path(user_path:Optional[Union[str,Path]])->str:
    if user_path and Path(user_path).exists():
        return str(user_path)
    for cand in _DEFAULT_SF_CANDIDATES:
        if cand.exists(): return str(cand)
    raise FileNotFoundError("No .sf2 SoundFont found. Place one in assets/sf2/.")

def _render_with_fluidsynth(pm,sf2_path,sr):
    sig=inspect.signature(pm.fluidsynth); params=sig.parameters
    kwargs={}
    if "sample_rate" in params: kwargs["sample_rate"]=sr
    elif "samplerate" in params: kwargs["samplerate"]=sr
    elif "fs" in params: kwargs["fs"]=sr
    kwargs["sf2_path"]=sf2_path
    audio=pm.fluidsynth(**kwargs)
    return np.clip(np.asarray(audio,dtype=np.float32),-1,1)

# ---------------- Public API ----------------
def generate_audio(chords:List[str],
                   chord_dur:float=1.0,
                   tempo:int=DEFAULT_TEMPO,
                   soundfont_path=None,
                   program:int=DEFAULT_PROG,
                   octave:int=DEFAULT_OCT,
                   include_metronome:bool=False)->Tuple[bytes,List[str],List[str]]:
    if not chords: raise ValueError("No chords provided")
    sf2=_resolve_soundfont_path(soundfont_path)
    skipped,errors=[],[]
    pm=pretty_midi.PrettyMIDI(); inst=pretty_midi.Instrument(program=int(program))
    t=0.0
    for ch in chords:
        try: mids=chord_token_to_midinums(ch,octave)
        except Exception as e:
            skipped.append(ch); errors.append(f"{ch} ➜ {e}"); t+=chord_dur; continue
        for m in mids:
            inst.notes.append(pretty_midi.Note(velocity=96,pitch=m,start=t,end=t+chord_dur))
        t+=chord_dur
    pm.instruments.append(inst)
    audio=_render_with_fluidsynth(pm,sf2,SAMPLE_RATE)
    if include_metronome:
        click1=_sine_click(SAMPLE_RATE,MET_CLICK_MS,MET_CLICK_FREQ,MET_CLICK_GAIN_1)
        clicko=_sine_click(SAMPLE_RATE,MET_CLICK_MS,MET_CLICK_FREQ,MET_CLICK_GAIN)
        spb=60.0/tempo; total=pm.get_end_time(); n=int(np.ceil(total/spb))
        for b in range(n):
            idx=int(b*spb*SAMPLE_RATE)
            _overlay(audio,click1 if b%4==0 else clicko,idx)
    buf=io.BytesIO(); sf.write(buf,audio,SAMPLE_RATE,format="WAV")
    return buf.getvalue(),skipped,errors
