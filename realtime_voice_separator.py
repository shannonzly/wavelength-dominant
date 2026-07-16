"""
Real-time two-speaker separation and toggle, using ClearVoice MossFormer2_SS_16K.

Speaker A tracks the sustained-loudest voice (lecture-hall dominant heuristic);
Speaker B is the other separated stream. Toggle A/B freely while listening.
"""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly, stft, istft

MODEL_SR = 16000
# Keep windows short so MPS stays near real-time. If wall-clock for a hop
# exceeds HOP_SEC, we *stretch* the emit so headphones never starve (slight pitch drop).
# ClearVoice's high-level API pads every chunk to 2s and is far too slow live.
WINDOW_SEC = 0.5
HOP_SEC = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * MODEL_SR)
HOP_SAMPLES = int(HOP_SEC * MODEL_SR)
HOP_CENTER = (WINDOW_SAMPLES - HOP_SAMPLES) // 2  # 0 when window==hop
CROSSFADE_SAMPLES = int(0.09 * MODEL_SR)  # 90 ms equal-power seam (softens 0.5s resets)
BLOCK_SEC = 0.05  # 50 ms sounddevice blocks — steadier I/O
# Larger ready + low-water absorb MPS jitter without silence gaps.
PLAYBACK_READY_SEC = 1.25
PLAYBACK_LOW_WATER_SEC = 0.5
PLAYBACK_HIGH_WATER_SEC = 2.0  # max keep in local playback buf
MAX_INPUT_BUFFER = WINDOW_SAMPLES + HOP_SAMPLES * 8
TRACK_TAIL_SEC = 0.25
TRACK_TAIL = int(TRACK_TAIL_SEC * MODEL_SR)
LEVEL_RATIO_MAX = 1.15  # tight hop-to-hop gain limit after separation
OUTPUT_GAIN = 1.15  # mild lift only — 2.5x was blasting BT buds
OUTPUT_CEILING = 0.7  # hard peak limit after gain
OUTPUT_LATENCY = 0.25  # seconds — helps Bluetooth devices
# Pitch-stretch and grain-loop pads both hurt clarity; rely on buffer instead.
ENABLE_TIME_STRETCH = False
ENABLE_DURATION_PAD = False
REALTIME_SLACK = 1.05
DURATION_PAD_CAP_SEC = 0.08
ENABLE_DENOISE = False  # STFT denoise costs time; caused extra underruns on MPS
MAX_OUT_QUEUE_SEC = 2.5

# Sticky singular-voice lock: once A is chosen, stay there. Only reassign if
# another stream is clearly louder *while A is also active* for a long stretch.
ENABLE_DOMINANT_REASSIGN = True
RMS_EMA_ALPHA = 0.08  # slow loudness memory — resists brief bursts
DOMINANT_RATIO_DB = 15.0  # B must beat A by this much (dB)
DOMINANT_HOLD_WINDOWS = 24  # ~12s at 0.5s hops — needs sustained evidence
QUIET_RMS = 1e-3
FP_UPDATE_MIN_RMS = 3e-3
# Weight continuity of A much higher than B when resolving permutation.
A_IDENTITY_WEIGHT = 4.0
SWAP_MARGIN = 0.85  # once locked, need a very clear match win to remap
# Hold locked A's playback level near a slow target (don't chase silence).
ENABLE_LOCK_LEVEL = True
LOCK_LEVEL_TARGET_ALPHA = 0.04  # slow target adaptation
LOCK_LEVEL_SPEECH_FLOOR = 2.5e-3  # below this: do not boost / retarget
LOCK_LEVEL_MAX_GAIN = 2.2
LOCK_LEVEL_MIN_GAIN = 0.4
# Soft-join consecutive playback chunks (covers residual seam ticks).
PLAYBACK_JOIN_SAMPLES = 128  # @ output rate, applied after reserving

MODEL_NAME = "MossFormer2_SS_16K"


def load_model():
    """Load ClearVoice speech separation model; return SpeechModel wrapper."""
    from clearvoice import ClearVoice

    print(
        f"Loading ClearVoice model: {MODEL_NAME} "
        "(first run downloads pretrained weights)..."
    )
    separator = ClearVoice(task="speech_separation", model_names=[MODEL_NAME])
    speech_model = separator.models[0]
    speech_model.model.eval()
    print(f"Model loaded on {speech_model.device}.")
    dummy = torch.zeros(1, WINDOW_SAMPLES, device=speech_model.device)
    with torch.inference_mode():
        _ = speech_model.model(dummy)
        if speech_model.device.type == "mps":
            torch.mps.synchronize()
    print("Warmup done.")
    return speech_model


class RateConverter:
    """Stream-friendly sample-rate converter using resample_poly."""

    def __init__(self, from_sr, to_sr):
        self.from_sr = int(round(from_sr))
        self.to_sr = int(round(to_sr))
        if self.from_sr <= 0 or self.to_sr <= 0:
            raise ValueError(f"invalid rates: {from_sr} -> {to_sr}")
        g = math.gcd(self.from_sr, self.to_sr)
        self.up = self.to_sr // g
        self.down = self.from_sr // g
        self._buf = np.zeros(0, dtype=np.float32)

    def convert(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.from_sr == self.to_sr:
            return x
        if x.size == 0:
            return x
        self._buf = np.concatenate([self._buf, x])
        n = (len(self._buf) // self.down) * self.down
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        chunk = self._buf[:n]
        self._buf = self._buf[n:]
        y = resample_poly(chunk, self.up, self.down)
        return np.asarray(y, dtype=np.float32)


def get_target_devices():
    """Prefer built-in mic + headphone/BT output; fall back to system defaults.

    BT buds often appear as *two* PortAudio devices (mic-only + stereo out).
    Only the entry with max_output_channels > 0 is usable for playback. If the
    buds are connected in HFP-only mode, PortAudio may list the mic but not
    the A2DP output — then we fall back and print available outputs.
    """
    devices = sd.query_devices()
    input_device_id = None
    output_device_id = None

    for idx, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0:
            if "built-in" in name or "macbook" in name or "imac" in name or "internal" in name:
                input_device_id = idx
                print(f"Found Mac Built-in Mic: '{d['name']}' (ID: {idx})")
                break

    if input_device_id is None:
        input_device_id = sd.default.device[0]
        print(f"No explicit built-in mic found. Using system default input (ID: {input_device_id})")

    headphone_keys = (
        "airpods",
        "beats",
        "bud",
        "solo",
        "headphone",
        "headset",
        "bluetooth",
        "wh-",  # Sony WH-
        "wf-",  # Sony WF-
        "sony",
        "bose",
        "studio",
        "earphone",
        "a2dp",
    )
    skip_keys = ("zoom", "virtual", "blackhole", "aggregate", "multi-output")

    def _is_skipped(name_l):
        return any(k in name_l for k in skip_keys)

    def _is_headphone(name_l):
        if _is_skipped(name_l):
            return False
        return any(k in name_l for k in headphone_keys)

    def _is_builtin_speaker(name_l):
        if "speaker" not in name_l:
            return False
        return any(k in name_l for k in ("macbook", "built-in", "imac", "internal", "mac mini", "mac pro"))

    # All real output endpoints (for debugging + fallback).
    outputs = [
        (idx, d)
        for idx, d in enumerate(devices)
        if d["max_output_channels"] > 0 and not _is_skipped(d["name"].lower())
    ]
    headphone_outs = [
        (idx, d) for idx, d in outputs if _is_headphone(d["name"].lower())
    ]

    default_out = sd.default.device[1]
    default_name = (
        devices[default_out]["name"]
        if default_out is not None and 0 <= default_out < len(devices)
        else "?"
    )
    default_l = default_name.lower()

    # 1) System default already pointed at buds/headphones (best signal).
    if default_out is not None and _is_headphone(default_l):
        output_device_id = default_out
        print(f"Found headphones output (system default): '{default_name}' (ID: {default_out})")
    # 2) Named match among stereo/BT outputs (Beats Solo Buds out=#4, not mic=#3).
    elif headphone_outs:
        output_device_id, d = headphone_outs[0]
        print(f"Found headphones output: '{d['name']}' (ID: {output_device_id})")
    # 3) Default is some non-builtin device we don't name-match yet.
    elif default_out is not None and not _is_builtin_speaker(default_l) and not _is_skipped(default_l):
        output_device_id = default_out
        print(
            f"No name match for headphones; using non-builtin system default "
            f"output: '{default_name}' (ID: {default_out})"
        )
    else:
        output_device_id = default_out
        print(
            f"No headphones match. Using system default output: "
            f"'{default_name}' (ID: {output_device_id})"
        )
        if outputs:
            listing = ", ".join(f"#{i} '{d['name']}'" for i, d in outputs)
            print(f"Available outputs: {listing}")
        # Hint when buds show up as mic-only (no A2DP playback device yet).
        bt_mics = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0
            and d["max_output_channels"] == 0
            and _is_headphone(d["name"].lower())
        ]
        if bt_mics and not headphone_outs:
            print(
                "Note: BT mic visible but no BT stereo output — select the buds "
                f"as Mac sound output ({bt_mics[0]}), then restart listening."
            )

    print(
        f"Routing: mic #{input_device_id} → headphones/speakers "
        f"#{output_device_id} ({devices[output_device_id]['name']})"
    )

    return input_device_id, output_device_id


def _boost_output(x):
    """Gentle gain + peak ceiling — never force quiet frames up to a target peak."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x = x * OUTPUT_GAIN
    peak = float(np.max(np.abs(x)) + 1e-8)
    if peak > OUTPUT_CEILING:
        x = x * (OUTPUT_CEILING / peak)
    return x.astype(np.float32)


def _test_tone(sr, sec=0.3, freq=880.0, amp=0.25):
    """Short beep to verify the output device is actually audible."""
    n = max(1, int(sr * sec))
    t = np.arange(n, dtype=np.float32) / float(sr)
    env = np.linspace(0.0, 1.0, min(n, int(0.02 * sr)), dtype=np.float32)
    tone = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    tone[: len(env)] *= env
    tone[-len(env) :] *= env[::-1]
    return tone


def _device_sr(device_id):
    info = sd.query_devices(device_id)
    return int(round(info["default_samplerate"]))


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def _corr(x, y):
    n = min(len(x), len(y))
    if n < 16:
        return 0.0
    a = np.asarray(x[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da < 1e-8 or db < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _spectral_fingerprint(x, n_fft=512):
    """Mean log-magnitude spectrum — used to keep speaker identity stable."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    spec = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.log1p(np.abs(spec).astype(np.float64))
    mag -= mag.mean()
    norm = np.linalg.norm(mag) + 1e-8
    return (mag / norm).astype(np.float32)


def _fp_sim(a, b):
    if a is None or b is None:
        return 0.0
    n = min(len(a), len(b))
    return float(np.dot(a[:n], b[:n]))


class SpeakerTracker:
    """Keep A/B identity across windows; A locks onto one sustained voice.

    1) Resolve MossFormer2 permutation via corr + spectral fingerprints,
       heavily biased toward keeping the locked A identity.
    2) Only reassign A→B when B is clearly louder *while A is also speaking*
       for an extended streak — pauses / brief Q&A never steal the lock.
    """

    def __init__(self):
        self.prev_a = None
        self.prev_b = None
        self.fp_a = None
        self.fp_b = None
        self.locked = False
        self.rms_a = None
        self.rms_b = None
        self._dominant_streak = 0  # consecutive windows of strong B dominance

    def align(self, src0, src1):
        r0, r1 = _rms(src0), _rms(src1)

        # First window: louder → A by convention (stable starting point).
        if self.prev_a is None:
            if r0 >= r1:
                stream_a, stream_b = src0, src1
            else:
                stream_a, stream_b = src1, src0
            self._update(stream_a, stream_b)
            self.locked = True
            return stream_a, stream_b

        # Both near silence: freeze dominant logic / RMS, but still map identity.
        if r0 < QUIET_RMS and r1 < QUIET_RMS:
            stream_a, stream_b = self._match_to_prev(src0, src1)
            self._update(stream_a, stream_b, update_rms=False)
            self._dominant_streak = 0
            return stream_a, stream_b

        # If one output is basically silent, keep identity via prev match —
        # do NOT map "whatever is louder" onto A (that hands A to residual
        # whenever the main talker pauses). Never count toward reassign.
        quiet_ratio = 0.12
        if r0 < quiet_ratio * max(r1, 1e-6) or r1 < quiet_ratio * max(r0, 1e-6):
            stream_a, stream_b = self._match_to_prev(src0, src1)
            self._update(stream_a, stream_b)
            self._dominant_streak = 0
            return stream_a, stream_b

        stream_a, stream_b = self._match_to_prev(src0, src1)
        self._update(stream_a, stream_b)
        if ENABLE_DOMINANT_REASSIGN:
            stream_a, stream_b = self._maybe_reassign_dominant()
        return stream_a, stream_b

    def _match_to_prev(self, src0, src1, allow_swap=True):
        """Map raw sources to previous A/B via corr+fingerprint (A weighted)."""
        fp0 = _spectral_fingerprint(src0)
        fp1 = _spectral_fingerprint(src1)
        tail0, tail1 = src0[-TRACK_TAIL:], src1[-TRACK_TAIL:]
        prev_a, prev_b = self.prev_a[-TRACK_TAIL:], self.prev_b[-TRACK_TAIL:]
        w = A_IDENTITY_WEIGHT
        score_keep = (
            w * (_corr(tail0, prev_a) + _fp_sim(fp0, self.fp_a))
            + (_corr(tail1, prev_b) + _fp_sim(fp1, self.fp_b))
        )
        score_swap = (
            w * (_corr(tail1, prev_a) + _fp_sim(fp1, self.fp_a))
            + (_corr(tail0, prev_b) + _fp_sim(fp0, self.fp_b))
        )
        # Once locked: only remap if swap is clearly better *and* the candidate
        # for A actually resembles locked A (not residual / other talker).
        margin = SWAP_MARGIN if (self.locked and allow_swap) else 0.0
        if allow_swap and score_swap > score_keep + margin:
            if self.locked:
                cand_a_fp = fp1
                if _fp_sim(cand_a_fp, self.fp_a) + 0.05 < _fp_sim(fp0, self.fp_a):
                    # Candidate would break A identity — keep mapping.
                    return src0, src1
            return src1, src0
        return src0, src1

    def _maybe_reassign_dominant(self):
        """Swap A/B only under extended, strong evidence that B is the new main.

        Requires: both voices active this window, B's EMA clearly louder than A's
        held loudness, B spectrally unlike locked A, for many consecutive windows.
        A pause + someone else talking does *not* count.
        """
        if self.rms_a is None or self.rms_b is None:
            return self.prev_a, self.prev_b

        ra = _rms(self.prev_a)
        rb = _rms(self.prev_b)
        both_active = ra >= FP_UPDATE_MIN_RMS and rb >= FP_UPDATE_MIN_RMS

        ratio_db = 20.0 * math.log10((self.rms_b + 1e-12) / (self.rms_a + 1e-12))

        # Candidate B should not be a near-copy of locked A (residual bleed).
        fp_b_now = _spectral_fingerprint(self.prev_b)
        looks_like_a = _fp_sim(fp_b_now, self.fp_a)
        looks_like_b = _fp_sim(fp_b_now, self.fp_b)
        is_other_voice = looks_like_b >= looks_like_a + 0.12

        if (
            both_active
            and ratio_db >= DOMINANT_RATIO_DB
            and is_other_voice
        ):
            self._dominant_streak += 1
        else:
            # Decay streak slowly so one ambiguous frame doesn't wipe progress,
            # but short Q&A still cannot accumulate a full reassign.
            self._dominant_streak = max(0, self._dominant_streak - 2)

        if self._dominant_streak >= DOMINANT_HOLD_WINDOWS:
            stream_a, stream_b = self.prev_b, self.prev_a
            self.prev_a, self.prev_b = stream_a, stream_b
            self.fp_a, self.fp_b = self.fp_b, self.fp_a
            self.rms_a, self.rms_b = self.rms_b, self.rms_a
            self._dominant_streak = 0
            print(
                f"Dominant reassigned → A "
                f"(B led by {ratio_db:.1f} dB for {DOMINANT_HOLD_WINDOWS} windows)"
            )
            return self.prev_a, self.prev_b

        return self.prev_a, self.prev_b

    def _update(self, stream_a, stream_b, update_rms=True):
        self.prev_a = stream_a.astype(np.float32, copy=True)
        self.prev_b = stream_b.astype(np.float32, copy=True)
        fp_a = _spectral_fingerprint(stream_a)
        fp_b = _spectral_fingerprint(stream_b)
        alpha = 0.15  # slow fingerprint drift once locked
        ra, rb = _rms(stream_a), _rms(stream_b)
        if self.fp_a is None:
            self.fp_a, self.fp_b = fp_a, fp_b
        else:
            # Only update A's fingerprint when A is clearly active — never let
            # a pause + residual rewrite the locked talker identity.
            if ra >= FP_UPDATE_MIN_RMS and ra >= rb * 0.85:
                self.fp_a = (1 - alpha) * self.fp_a + alpha * fp_a
                self.fp_a /= np.linalg.norm(self.fp_a) + 1e-8
            if rb >= FP_UPDATE_MIN_RMS and rb >= ra * 0.5:
                self.fp_b = (1 - alpha) * self.fp_b + alpha * fp_b
                self.fp_b /= np.linalg.norm(self.fp_b) + 1e-8

        if update_rms:
            if self.rms_a is None:
                self.rms_a, self.rms_b = ra, rb
            else:
                a = RMS_EMA_ALPHA
                # Hold A's loudness through pauses so B talking alone never
                # looks "dominant" just because A went quiet.
                if ra >= FP_UPDATE_MIN_RMS:
                    self.rms_a = (1 - a) * self.rms_a + a * ra
                if rb >= FP_UPDATE_MIN_RMS:
                    self.rms_b = (1 - a) * self.rms_b + a * rb


class ResidualDenoiser:
    """STFT residual denoise — n_fft=512, hop=128, strength=1.1, gate -48 dB."""

    def __init__(self, n_fft=512, hop=128, strength=1.1, noise_gate_db=-48.0):
        self.n_fft = n_fft
        self.hop = hop
        self.strength = strength
        self.noise_gate = 10 ** (noise_gate_db / 20.0)
        self.noise_mag = None

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size < self.n_fft:
            return x

        _, _, z = stft(
            x,
            fs=MODEL_SR,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            boundary="zeros",
            padded=True,
        )
        mag = np.abs(z)
        frame_e = mag.mean(axis=0)
        speechish = frame_e > max(np.median(frame_e) * 1.2, 1e-5)

        if np.any(~speechish):
            noise_est = mag[:, ~speechish].mean(axis=1)
        else:
            noise_est = np.percentile(mag, 20, axis=1)

        if self.noise_mag is None:
            self.noise_mag = noise_est
        else:
            self.noise_mag = 0.95 * self.noise_mag + 0.05 * noise_est

        noise = self.noise_mag[:, None]
        mask = 1.0 - self.strength * (noise / (mag + 1e-6))
        mask = np.clip(mask, 0.25, 1.0)
        quiet = frame_e < (np.median(frame_e) * 0.45)
        mask[:, quiet] *= 0.55

        z_hat = z * mask
        _, y = istft(
            z_hat,
            fs=MODEL_SR,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            input_onesided=True,
        )
        y = y[: len(x)].astype(np.float32)

        # Soft gate — avoid near-zero slam that clicks at hop edges
        r = _rms(y)
        if r < self.noise_gate:
            y *= max(0.2, r / (self.noise_gate + 1e-12))
        return y


def _seam_hops(prev_tail, hop, fade_len):
    """Equal-power crossfade + edge continuity; emit always full hop length.

    Holds a copy of the last `fade_len` samples for the *next* blend, but still
    emits the full hop so production stays near realtime (withholding fade
    samples each hop caused systematic underruns).
    """
    hop = np.asarray(hop, dtype=np.float32).reshape(-1).copy()
    if fade_len <= 0 or hop.size < fade_len * 2:
        return hop, None

    fade_len = min(fade_len, hop.size // 2)

    if prev_tail is not None:
        prev = np.asarray(prev_tail, dtype=np.float32).reshape(-1)
        fade_len = min(fade_len, len(prev), len(hop))
        head = hop[:fade_len].copy()

        if fade_len >= 2 and len(prev) >= 2:
            prev_end = float(prev[-1])
            prev_slope = float(prev[-1] - prev[-2])
            hop_start = float(head[0])
            hop_slope = float(head[1] - head[0])
            dc = prev_end - hop_start
            slope_err = prev_slope - hop_slope
            ramp = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            head = head + dc * ramp
            if abs(slope_err) > 1e-8:
                t = np.arange(fade_len, dtype=np.float32)
                head = head + (slope_err * t * ramp * 0.35)

        t = np.linspace(0.0, np.pi / 2.0, fade_len, dtype=np.float32)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        hop[:fade_len] = prev[-fade_len:] * fade_out + head * fade_in

    new_tail = hop[-fade_len:].copy()
    return hop, new_tail


def _soft_join(buf, nxt, fade_n=PLAYBACK_JOIN_SAMPLES):
    """Equal-power join when appending a new chunk to the playback buffer."""
    if buf is None or len(buf) == 0:
        return np.asarray(nxt, dtype=np.float32).reshape(-1)
    if nxt is None or len(nxt) == 0:
        return np.asarray(buf, dtype=np.float32).reshape(-1)
    buf = np.asarray(buf, dtype=np.float32).reshape(-1)
    nxt = np.asarray(nxt, dtype=np.float32).reshape(-1)
    fade_n = int(min(fade_n, len(buf), len(nxt)))
    if fade_n < 4:
        return np.concatenate([buf, nxt])
    t = np.linspace(0.0, np.pi / 2.0, fade_n, dtype=np.float32)
    mid = buf[-fade_n:] * np.cos(t) + nxt[:fade_n] * np.sin(t)
    return np.concatenate([buf[:-fade_n], mid, nxt[fade_n:]]).astype(np.float32)


def _pad_hop_to_wall(hop, wall_s, cap_sec=DURATION_PAD_CAP_SEC, slack=REALTIME_SLACK):
    """Extend hop with a looped grain (ping-pong) so emit covers wall time.

    Silence pads caused audible breakup every slow frame. Looping the hop's
    last ~40–80ms with an equal-power junction keeps continuous sound without
    pitch-shifting the whole utterance.
    """
    hop = np.asarray(hop, dtype=np.float32).reshape(-1)
    if hop.size < 8:
        return hop
    hop_sec = hop.size / float(MODEL_SR)
    target_sec = max(hop_sec, float(wall_s) * float(slack))
    deficit_sec = target_sec - hop_sec
    if deficit_sec <= 0.001:
        return hop
    pad_n = int(round(min(deficit_sec, cap_sec) * MODEL_SR))
    if pad_n <= 0:
        return hop

    grain_n = min(hop.size, max(CROSSFADE_SAMPLES, int(0.05 * MODEL_SR)))
    grain = hop[-grain_n:].copy()
    period = np.concatenate([grain, grain[::-1]])
    reps = int(math.ceil(pad_n / float(len(period)))) + 1
    tiled = np.tile(period, reps)

    fade = min(CROSSFADE_SAMPLES, hop.size, pad_n)
    if fade <= 0:
        return np.concatenate([hop, tiled[:pad_n]]).astype(np.float32)

    # Keep full hop+pad length; equal-power blend only the junction samples.
    out = np.concatenate([hop, tiled[:pad_n]]).astype(np.float32)
    t = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    ju = hop.size - fade
    out[ju:hop.size] = hop[-fade:] * np.cos(t) + tiled[:fade] * np.sin(t)
    return out


def _match_hop_level(chunk, prev_rms, max_ratio=LEVEL_RATIO_MAX):
    """Limit sudden loudness jumps between hops (separation scale flicker)."""
    chunk = np.asarray(chunk, dtype=np.float32)
    cur = _rms(chunk)
    if prev_rms is None or cur < 1e-8 or prev_rms < 1e-8:
        return chunk, cur
    ratio = float(prev_rms / cur)
    if ratio > max_ratio:
        chunk = chunk * max_ratio
        cur = _rms(chunk)
    elif ratio < 1.0 / max_ratio:
        chunk = chunk * (1.0 / max_ratio)
        cur = _rms(chunk)
    return chunk, cur


def _stabilize_lock_level(chunk, target_rms):
    """Hold locked-voice hops near a slow target RMS (skip silence).

    Returns (chunk, new_target_rms, post_rms).
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    cur = _rms(chunk)
    if cur < LOCK_LEVEL_SPEECH_FLOOR:
        return chunk, target_rms, cur

    if target_rms is None:
        return chunk, cur, cur

    gain = float(target_rms / (cur + 1e-12))
    gain = max(LOCK_LEVEL_MIN_GAIN, min(LOCK_LEVEL_MAX_GAIN, gain))
    chunk = (chunk * gain).astype(np.float32)
    post = _rms(chunk)
    # Nudge target only from real speech so level stays flat across phrases.
    a = LOCK_LEVEL_TARGET_ALPHA
    new_target = (1.0 - a) * target_rms + a * post
    return chunk, new_target, post


def _fit_hop_to_realtime(hop, wall_s, slack=REALTIME_SLACK):
    """Stretch hop so its duration covers wall_s * slack (prevents underruns).

    When the model is slightly slower than real-time, a tiny slowdown/pitch drop
    is far better than silent rebuffering.
    """
    hop = np.asarray(hop, dtype=np.float32).reshape(-1)
    if hop.size < 8:
        return hop
    target_sec = max(HOP_SEC, float(wall_s) * slack)
    target_n = int(round(target_sec * MODEL_SR))
    target_n = max(target_n, hop.size)
    # Cap stretch so speech does not become absurdly slow if a frame stalls.
    max_n = int(round(HOP_SEC * 1.8 * MODEL_SR))
    target_n = min(target_n, max_n)
    if target_n == hop.size:
        return hop
    g = math.gcd(target_n, hop.size)
    up, down = target_n // g, hop.size // g
    y = resample_poly(hop, up, down)
    y = np.asarray(y, dtype=np.float32)
    if len(y) > target_n:
        y = y[:target_n]
    elif len(y) < target_n:
        y = np.pad(y, (0, target_n - len(y)))
    return y


def _separate_window(speech_model, window, mix_peak):
    """Direct model forward — skips ClearVoice's 2s padding decode path."""
    device = speech_model.device
    window_norm = (window / mix_peak).astype(np.float32)
    audio = torch.from_numpy(window_norm[None, :]).to(device)
    with torch.inference_mode():
        out_list = speech_model.model(audio)
        if device.type == "mps":
            torch.mps.synchronize()
    src0 = out_list[0][0].detach().cpu().numpy().astype(np.float32)
    src1 = out_list[1][0].detach().cpu().numpy().astype(np.float32)
    # Restore input scale only — do NOT force each source to mix RMS
    # (that was amplifying residual noise on quiet channels).
    src0 *= mix_peak
    src1 *= mix_peak
    return src0, src1


class AudioApp:
    def __init__(self):
        self.speech_model = load_model()
        self.tracker = SpeakerTracker()
        self.denoise_a = ResidualDenoiser()
        self.denoise_b = ResidualDenoiser()

        self.selected = "A"
        self.selected_lock = threading.Lock()

        self.raw_queue = queue.Queue()
        self.out_queue = queue.Queue(maxsize=200)
        self._playback_buf = np.zeros(0, dtype=np.float32)
        self._playback_lock = threading.Lock()

        self.running = False
        self.listening = False
        self.in_stream = None
        self.out_stream = None
        self.proc_thread = None
        self.playback_ready = False

        self.input_sr = MODEL_SR
        self.output_sr = MODEL_SR
        self.in_resampler = RateConverter(MODEL_SR, MODEL_SR)
        self.out_resampler = RateConverter(MODEL_SR, MODEL_SR)

        self._model_sr_buffer = np.zeros(0, dtype=np.float32)
        self._fade_tail_a = None
        self._fade_tail_b = None
        self._hop_rms_a = None
        self._hop_rms_b = None
        self._lock_target_rms = None  # slow target level for locked A
        self._last_out_sample = 0.0

        self.stats_lock = threading.Lock()
        self.stats = {
            "in_rms": 0.0,
            "out_rms": 0.0,
            "infer_s": 0.0,
            "windows": 0,
            "swaps": 0,
            "underruns": 0,
            "buf_s": 0.0,
            "queued": 0,
        }

    def input_callback(self, indata, frames, time_info, status):
        if status:
            print("input status:", status)
        mono = indata[:, 0].copy()
        self.raw_queue.put(mono)

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            print("output status:", status)
        with self._playback_lock:
            need = int(self.output_sr * PLAYBACK_READY_SEC)
            low_water = int(self.output_sr * PLAYBACK_LOW_WATER_SEC)
            high_water = int(self.output_sr * PLAYBACK_HIGH_WATER_SEC)

            # Warm-up: fill to ready. After start: keep a healthy local buffer.
            if not self.playback_ready:
                pull_target = max(need, frames * 8)
            else:
                pull_target = max(low_water, frames * 8)
            while len(self._playback_buf) < pull_target:
                try:
                    nxt = self.out_queue.get_nowait()
                    self._playback_buf = _soft_join(self._playback_buf, nxt)
                except queue.Empty:
                    break
            # Cap runaway local buffer (extras stay in out_queue).
            if len(self._playback_buf) > high_water:
                # Keep newest high_water samples? Prefer keep front (next to play).
                self._playback_buf = self._playback_buf[:high_water]

            with self.stats_lock:
                self.stats["buf_s"] = len(self._playback_buf) / float(self.output_sr)
                self.stats["queued"] = self.out_queue.qsize()

            if not self.playback_ready:
                if len(self._playback_buf) >= need:
                    self.playback_ready = True
                    print(
                        f"Playback started "
                        f"({len(self._playback_buf) / self.output_sr:.2f}s buffered)."
                    )
                else:
                    outdata.fill(0.0)
                    return

            if len(self._playback_buf) >= frames:
                chunk = self._playback_buf[:frames]
                self._playback_buf = self._playback_buf[frames:]
            else:
                # Soft shortfall: play remainder, hold last sample briefly (not mute).
                with self.stats_lock:
                    self.stats["underruns"] += 1
                have = len(self._playback_buf)
                chunk = np.zeros(frames, dtype=np.float32)
                if have:
                    chunk[:have] = self._playback_buf
                    last = float(self._playback_buf[-1])
                    self._playback_buf = np.zeros(0, dtype=np.float32)
                else:
                    last = float(self._last_out_sample)
                # Soft hold then decay — continuous enough to hide BT/MPS jitter.
                hold = min(frames - have, max(64, frames // 4))
                if hold > 0:
                    env = np.linspace(1.0, 0.15, hold, dtype=np.float32)
                    chunk[have : have + hold] = last * env
                queue_empty = self.out_queue.empty()
                if have == 0 and queue_empty:
                    self.playback_ready = False
                    print(
                        f"output underrun — rebuffering "
                        f"(played partial {have}, need {need})"
                    )
                else:
                    print(
                        f"output soft shortfall — keeping stream "
                        f"(have={have}, queued≈{self.out_queue.qsize()}, "
                        f"need_block={frames})"
                    )

            if chunk.size:
                self._last_out_sample = float(chunk[-1])

            with self.stats_lock:
                self.stats["buf_s"] = len(self._playback_buf) / float(self.output_sr)

        outdata[:, 0] = chunk
        if outdata.shape[1] > 1:
            outdata[:, 1] = chunk

    def _emit_separated_hops(self, hop_a, hop_b, wall_s, infer_s, swapped):
        """Level-match, grain-pad if slow, seam, boost, and enqueue for playback."""
        if ENABLE_TIME_STRETCH:
            hop_a = _fit_hop_to_realtime(hop_a, wall_s)
            hop_b = _fit_hop_to_realtime(hop_b, wall_s)
        elif ENABLE_DURATION_PAD and wall_s > HOP_SEC:
            # Pad *before* seam so fade tails include the pad — no gap after crossfade.
            hop_a = _pad_hop_to_wall(hop_a, wall_s)
            hop_b = _pad_hop_to_wall(hop_b, wall_s)

        hop_a, self._hop_rms_a = _match_hop_level(hop_a, self._hop_rms_a)
        hop_b, self._hop_rms_b = _match_hop_level(hop_b, self._hop_rms_b)

        # Keep locked dominant voice at a steady listening level.
        if ENABLE_LOCK_LEVEL:
            hop_a, self._lock_target_rms, self._hop_rms_a = _stabilize_lock_level(
                hop_a, self._lock_target_rms
            )

        new_a, self._fade_tail_a = _seam_hops(
            self._fade_tail_a, hop_a, CROSSFADE_SAMPLES
        )
        new_b, self._fade_tail_b = _seam_hops(
            self._fade_tail_b, hop_b, CROSSFADE_SAMPLES
        )

        with self.selected_lock:
            chosen = new_a if self.selected == "A" else new_b

        chosen = _boost_output(chosen)
        out_chunk = self.out_resampler.convert(chosen.astype(np.float32))
        with self.stats_lock:
            self.stats["infer_s"] = infer_s
            self.stats["windows"] += 1
            if swapped:
                self.stats["swaps"] += 1
            self.stats["out_rms"] = _rms(out_chunk) if out_chunk.size else 0.0
            buf_s = self.stats.get("buf_s", 0.0)

        if out_chunk.size == 0:
            return

        emit_sec = len(out_chunk) / float(self.output_sr)
        if wall_s > HOP_SEC:
            if ENABLE_TIME_STRETCH:
                mode = "stretched"
            elif ENABLE_DURATION_PAD:
                mode = "grain-pad"
            else:
                mode = "natural"
            print(
                f"slow frame: wall={wall_s:.2f}s infer={infer_s:.2f}s "
                f"emit={emit_sec:.2f}s buf={buf_s:.2f}s ({mode})"
            )

        try:
            self.out_queue.put(out_chunk, timeout=0.5)
        except queue.Full:
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(out_chunk)
            except queue.Full:
                pass

    def _process_available_windows(self, max_windows=3):
        """Run separation on buffered windows until empty or output queue is full enough."""
        processed = 0
        while processed < max_windows and len(self._model_sr_buffer) >= WINDOW_SAMPLES:
            # Avoid unbounded latency if playback is already well ahead.
            if self.out_queue.qsize() * HOP_SEC >= MAX_OUT_QUEUE_SEC:
                break

            window = self._model_sr_buffer[:WINDOW_SAMPLES]
            self._model_sr_buffer = self._model_sr_buffer[HOP_SAMPLES:]

            mix_peak = float(np.max(np.abs(window)) + 1e-8)

            t0 = time.perf_counter()
            src0, src1 = _separate_window(self.speech_model, window, mix_peak)
            infer_s = time.perf_counter() - t0

            n = min(len(src0), len(src1), WINDOW_SAMPLES)
            src0 = src0[:n]
            src1 = src1[:n]
            if n < WINDOW_SAMPLES:
                pad = WINDOW_SAMPLES - n
                src0 = np.pad(src0, (0, pad))
                src1 = np.pad(src1, (0, pad))

            prev_fp_a = None if self.tracker.fp_a is None else self.tracker.fp_a.copy()
            stream_a, stream_b = self.tracker.align(src0, src1)
            swapped = (
                prev_fp_a is not None
                and _fp_sim(_spectral_fingerprint(stream_a), prev_fp_a)
                < _fp_sim(_spectral_fingerprint(stream_b), prev_fp_a)
            )

            if ENABLE_DENOISE:
                stream_a = self.denoise_a.process(stream_a)
                stream_b = self.denoise_b.process(stream_b)
            hop_a = stream_a[HOP_CENTER : HOP_CENTER + HOP_SAMPLES].astype(np.float32)
            hop_b = stream_b[HOP_CENTER : HOP_CENTER + HOP_SAMPLES].astype(np.float32)

            wall_s = time.perf_counter() - t0
            self._emit_separated_hops(hop_a, hop_b, wall_s, infer_s, swapped)
            processed += 1

            # Fold mic that arrived during this infer so the next window is current.
            while True:
                try:
                    raw_more = self.raw_queue.get_nowait()
                except queue.Empty:
                    break
                more = self.in_resampler.convert(raw_more)
                if more.size:
                    self._model_sr_buffer = np.concatenate(
                        [self._model_sr_buffer, more]
                    )
            while len(self._model_sr_buffer) > MAX_INPUT_BUFFER:
                self._model_sr_buffer = self._model_sr_buffer[HOP_SAMPLES:]

            # One window per cycle keeps A/B tracking stable; backlog drains
            # on subsequent mic blocks instead of burst-flipping identity.
            break

        return processed

    def processing_loop(self):
        while self.running:
            try:
                raw = self.raw_queue.get(timeout=1)
            except queue.Empty:
                # Still try to process any leftover buffered audio.
                if len(self._model_sr_buffer) >= WINDOW_SAMPLES:
                    self._process_available_windows(max_windows=1)
                continue

            model_audio = self.in_resampler.convert(raw)
            if model_audio.size == 0:
                continue

            with self.stats_lock:
                self.stats["in_rms"] = _rms(model_audio)

            self._model_sr_buffer = np.concatenate([self._model_sr_buffer, model_audio])

            # Prefer a bit of latency over dropping hops mid-phrase.
            while len(self._model_sr_buffer) > MAX_INPUT_BUFFER:
                self._model_sr_buffer = self._model_sr_buffer[HOP_SAMPLES:]

            self._process_available_windows(max_windows=1)

    def set_selected(self, choice):
        with self.selected_lock:
            self.selected = choice

    def _open_streams(self, input_device, output_device):
        self.input_sr = _device_sr(input_device)
        self.output_sr = _device_sr(output_device)
        self.in_resampler = RateConverter(self.input_sr, MODEL_SR)
        self.out_resampler = RateConverter(MODEL_SR, self.output_sr)

        print(
            f"Audio: input #{input_device} @ {self.input_sr} Hz → "
            f"model {MODEL_SR} Hz → output #{output_device} @ {self.output_sr} Hz"
        )

        in_blocksize = max(64, int(self.input_sr * BLOCK_SEC))
        out_blocksize = max(64, int(self.output_sr * BLOCK_SEC))

        out_info = sd.query_devices(device=output_device)
        out_channels = 2 if out_info["max_output_channels"] >= 2 else 1

        in_stream = sd.InputStream(
            device=input_device,
            samplerate=self.input_sr,
            channels=1,
            dtype="float32",
            callback=self.input_callback,
            blocksize=in_blocksize,
        )
        out_stream = sd.OutputStream(
            device=output_device,
            samplerate=self.output_sr,
            channels=out_channels,
            dtype="float32",
            callback=self.output_callback,
            blocksize=out_blocksize,
            latency=OUTPUT_LATENCY,
        )
        return in_stream, out_stream

    def start_streams(self):
        if self.listening:
            return

        self.playback_ready = False
        self.tracker = SpeakerTracker()
        self.denoise_a = ResidualDenoiser()
        self.denoise_b = ResidualDenoiser()
        self._model_sr_buffer = np.zeros(0, dtype=np.float32)
        self._fade_tail_a = None
        self._fade_tail_b = None
        self._hop_rms_a = None
        self._hop_rms_b = None
        self._lock_target_rms = None
        self._last_out_sample = 0.0

        with self.stats_lock:
            self.stats["underruns"] = 0
            self.stats["windows"] = 0
            self.stats["swaps"] = 0
            self.stats["buf_s"] = 0.0

        with self._playback_lock:
            self._playback_buf = np.zeros(0, dtype=np.float32)
        while not self.raw_queue.empty():
            try:
                self.raw_queue.get_nowait()
            except queue.Empty:
                break
        while not self.out_queue.empty():
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                break

        preferred_in, preferred_out = get_target_devices()
        candidates = [
            (preferred_in, preferred_out),
            (preferred_in, sd.default.device[1]),
            (sd.default.device[0], sd.default.device[1]),
        ]

        last_err = None
        in_stream = None
        out_stream = None
        for input_device, output_device in candidates:
            try:
                in_stream, out_stream = self._open_streams(input_device, output_device)
                in_stream.start()
                out_stream.start()
                self.in_stream = in_stream
                self.out_stream = out_stream
                break
            except Exception as err:
                last_err = err
                print(f"Failed to open devices in={input_device} out={output_device}: {err}")
                try:
                    if in_stream is not None:
                        in_stream.close()
                except Exception:
                    pass
                try:
                    if out_stream is not None:
                        out_stream.close()
                except Exception:
                    pass
                in_stream = None
                out_stream = None
        else:
            raise RuntimeError(f"Could not open audio streams: {last_err}") from last_err

        self.running = True
        self.listening = True
        self.proc_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.proc_thread.start()

        # Audible routing check — if you don't hear this beep, output device is wrong.
        # Do NOT force playback_ready; let the beep count toward the buffer so we
        # don't start draining before separated audio is queued.
        try:
            beep = _test_tone(self.output_sr, sec=0.25, freq=880.0, amp=0.3)
            self.out_queue.put_nowait(beep)
            print("Queued startup beep — speech follows once the buffer is warm.")
        except Exception as err:
            print(f"Could not queue startup beep: {err}")

    def stop(self):
        self.running = False
        self.listening = False
        try:
            if self.in_stream is not None:
                self.in_stream.stop()
                self.in_stream.close()
            if self.out_stream is not None:
                self.out_stream.stop()
                self.out_stream.close()
        except Exception:
            pass
        self.in_stream = None
        self.out_stream = None


def build_gui(app: AudioApp):
    root = tk.Tk()
    root.title("Wavelength - Dominant Speaker")
    root.geometry("460x360")

    status = tk.StringVar(value="Idle — press Start Listening")
    meters = tk.StringVar(value="mic: —   out: —   infer: —   buf: —   underrun:—")

    def _label_for(choice):
        return "Dominant (A)" if choice == "A" else "Other (B)"

    def choose(letter):
        if not app.listening:
            return
        app.set_selected(letter)
        status.set(f"Listening to: {_label_for(letter)}")

    label = ttk.Label(root, textvariable=status, font=("Helvetica", 14))
    label.pack(pady=10)

    meter_label = ttk.Label(root, textvariable=meters, font=("Helvetica", 11), foreground="gray")
    meter_label.pack(pady=4)

    def poll_stats():
        with app.stats_lock:
            s = dict(app.stats)
        if app.listening:
            meters.set(
                f"mic: {s['in_rms']:.3f}   out: {s['out_rms']:.3f}   "
                f"infer: {s['infer_s']:.2f}s   buf: {s.get('buf_s', 0):.2f}s   "
                f"q:{s.get('queued', 0)}   underrun:{s.get('underruns', 0)}"
            )
        root.after(200, poll_stats)

    def start_listening():
        start_btn.config(state=tk.DISABLED)
        try:
            app.start_streams()
        except Exception as err:
            start_btn.config(state=tk.NORMAL)
            status.set("Failed to start audio")
            messagebox.showerror("Audio Error", str(err))
            return
        btn_a.config(state=tk.NORMAL)
        btn_b.config(state=tk.NORMAL)
        status.set(
            f"Listening to: {_label_for(app.selected)} "
            "(A = locked dominant voice)"
        )

    start_btn = tk.Button(
        root,
        text="Start Listening",
        font=("Helvetica", 14),
        width=20,
        height=2,
        command=start_listening,
    )
    start_btn.pack(pady=6)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    btn_a = tk.Button(
        btn_frame,
        text="Dominant (A)",
        font=("Helvetica", 14),
        width=14,
        height=2,
        command=lambda: choose("A"),
        state=tk.DISABLED,
    )
    btn_a.grid(row=0, column=0, padx=10)

    btn_b = tk.Button(
        btn_frame,
        text="Other (B)",
        font=("Helvetica", 14),
        width=14,
        height=2,
        command=lambda: choose("B"),
        state=tk.DISABLED,
    )
    btn_b.grid(row=0, column=1, padx=10)

    note = ttk.Label(
        root,
        text="A locks onto one voice and holds level.\n"
        "B is the other stream — switch anytime.\n"
        "A only reassigns after a long stretch of clearly louder B.",
        justify="center",
        foreground="gray",
    )
    note.pack(pady=10)

    def on_close():
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(200, poll_stats)
    return root


def main():
    app = AudioApp()
    root = build_gui(app)
    root.mainloop()


if __name__ == "__main__":
    main()
