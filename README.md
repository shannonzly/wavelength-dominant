# Real-Time Two-Speaker Toggle with ClearVoice

PROTOTYPE -- DOMINANT VOICE AUTOLOCK

Run from computer

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

**Speaker A** locks onto the **loudest sustained** voice in the mix (lecture-hall
heuristic: usually the lecturer). **Speaker B** is the other separated stream.
You can switch A ↔ B anytime while listening.

## Requirements

- Python 3.9+
- A working microphone and speakers/headphones (**use headphones** — without
  them, the speaker output can feed back into the mic)
- On Linux, tkinter may need a separate package:
  `sudo apt install python3-tk`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python realtime_voice_separator.py
```

The first run downloads the pretrained model. After that it loads from cache.

## Using it

1. A small window opens with **Dominant (A)** and **Other (B)**.
2. Have two people speak into the mic (ideally close together, e.g. both
   near a laptop mic, or a single mic picking up both). In a lecture, aim
   the mic so the lecturer is clearly the loudest voice.
3. Click **Dominant (A)** to hear the loudest sustained talker, or **Other (B)**
   for the second stream. You can click back and forth freely while both
   people keep talking.

A starts on the louder voice, locks identity + listening level, and only
**reassigns** if another stream stays clearly louder *while A is also speaking*
for ~12 seconds — so pauses, questions, or coughs do not steal A.

## Important limitations

- **Latency**: ~1–2 seconds. MossFormer2 is not a causal/streaming model —
  it needs a short overlapping window of audio; playback also buffers ~0.75s
  so hops join smoothly without underruns.
- **Best for 1 strong speaker + intermittent other speech**: ideal for a
  lecturer with occasional Q&A. A hall full of many people talking at once
  will hurt quality on A as well — the model only outputs **two** sources.
- **Two speakers only**: `MossFormer2_SS_16K` is fixed at 2 sources. Extra
  voices tend to fold into noise or leak into A/B.
- **Speakers may still swap labels** after long silence or extreme level
  changes; if A no longer sounds like the dominant talker, try switching to B
  or restart listening.
- **Quality depends on your mic setup**: clean, close speech at 16 kHz works
  best; heavy noise, music, or strong reverb hurts results.
- **Performance**: MossFormer2 is heavier than smaller separators. On Apple
  Silicon, PyTorch MPS may help when available; otherwise CPU can fall
  behind the hop interval and you may hear dropouts.

## Files

- `realtime_voice_separator.py` — the app (run this)
- `requirements.txt` — pip dependencies
