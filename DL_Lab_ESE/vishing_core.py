# vishing_core.py
# Non-UI logic for the Real-Time Vishing Defence dashboard (realtime_demo.py).
# Kept separate from the Streamlit page so that it can be unit-tested
# (see test_vishing_core.py).
#
# The CNN class and log-mel front end are copied unchanged from
# vishing_experiments.py: the saved weights in cnn.pt only work with this exact
# architecture and these feature settings.
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

SR = 16000             # sample rate used in training (vishing_experiments.py: SR)
TRAIN_SECONDS = 4      # training clip length (vishing_experiments.py: LEN = 64000 samples)

# Shortest audio we will analyse. 1 s is the shortest window that was ever
# evaluated (results_chunk.csv). Below that there is no evaluation evidence at
# all, so shorter input is rejected rather than classified.
MIN_SECONDS = 1.0

# Score where the CNN's logit is 0. Accuracy and F1 in results_in_domain.csv and
# results_chunk.csv were computed at this point (prediction = score > 0.5). It
# is the only threshold with measured performance behind it.
DECISION_BOUNDARY = 0.5

# Bands from the original prototype. They were chosen by hand and have NOT been
# validated on any data. Kept only as labels, and always shown as heuristic.
HEURISTIC_BANDS = (0.4, 0.7)


class AudioError(ValueError):
    """The input cannot be analysed. The dashboard reports this instead of scoring it."""


class ModelLoadError(RuntimeError):
    """cnn.pt is missing, unreadable or does not match the CNN architecture."""


# ----------------------------------------------------------------------------- model
class CNN(nn.Module):  # must match the model trained in vishing_experiments.py
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.o = nn.Linear(64, 1)
    def forward(s, x): return s.o(s.f(x)).squeeze(1)


mel_t = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=64)
db = torchaudio.transforms.AmplitudeToDB()


def logmel(xb):
    m = db(mel_t(xb)).unsqueeze(1)
    return (m - m.mean((1, 2, 3), keepdim=True)) / (m.std((1, 2, 3), keepdim=True) + 1e-5)


def load_model(path):
    """Load cnn.pt from an explicit path. Raises ModelLoadError with a readable message."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"Model file not found: {path}\n"
            "Put cnn.pt (produced by vishing_experiments.py) in the same folder as realtime_demo.py.")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ModelLoadError(f"Could not read model file {path}: {type(e).__name__}: {e}") from e
    model = CNN()
    try:
        model.load_state_dict(state)
    except Exception as e:
        raise ModelLoadError(
            f"{path} does not match the CNN architecture expected by the dashboard: {e}") from e
    return model.eval()


# ----------------------------------------------------------------------------- audio
# Phase 5 input-validation rules (documented thresholds; the model, features and decision thresholds are unchanged):
#
# * Activity level: frame RMS (25 ms frames every 10 ms) of the mean-removed signal; the 95th percentile of those
#   frame levels, in dB relative to full scale (dBFS), is the clip's "activity level". The 95th percentile ignores a
#   few isolated clicks. A clip whose activity level is below MIN_ACTIVITY_DBFS = -60 dBFS has no audible content.
#   Why this matters: the CNN standardises every clip to zero mean / unit variance, so it rescales near-silent noise
#   into a "normal-looking" spectrogram and can return a confident but meaningless score (Phase 4: 1.00, HIGH).
#   Safety margin: the 1000 ASVspoof evaluation-subset clips and the 7 demo clips have activity levels between
#   -25.7 and -7.1 dBFS, i.e. at least 34 dB above the threshold (measured without using labels).
# * Constant / DC signal: if the activity level is below the threshold but the signal has a non-zero mean
#   (|mean| >= 1e-3), it is reported as a constant (DC) signal instead of "too quiet".
# * Truncation: established only where the file itself says how long it should be. WAV: the 'data' chunk declares
#   its size; fewer bytes present means truncation. FLAC: STREAMINFO declares the total samples; fewer decoded samples
#   means truncation. Truncated files are analysed on the audio that is present, with a visible warning. Other
#   damage (or WAV/FLAC headers that do not state a length) cannot be detected reliably and is not claimed.
# * Minimum duration stays 1 s (MIN_SECONDS), the shortest window ever evaluated (results_chunk.csv); there is no
#   evidence that shorter clips can be scored meaningfully (1 s windows already have 23 % EER).
MIN_ACTIVITY_DBFS = -60.0
LEVEL_FRAME, LEVEL_HOP = 400, 160
DC_MEAN_MIN = 1e-3


@dataclass
class AudioCheck:
    audio: np.ndarray            # mono float32 at 16 kHz, ready for make_windows
    orig_sr: int
    activity_dbfs: float
    warnings: list = field(default_factory=list)


def activity_level_dbfs(x, frame=LEVEL_FRAME, hop=LEVEL_HOP):
    """95th percentile of frame RMS (mean removed), in dBFS. -inf-like (-240) for a constant signal."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return float(20 * np.log10(max(np.percentile(rms, 95), 1e-12)))


def detect_truncation(data, decoded_frames, sr):
    """Return a warning string if the file's own header proves it is truncated, else None."""
    try:
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            pos, block_align = 12, None
            while pos + 8 <= len(data):
                cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
                if cid == b"fmt " and pos + 8 + 14 <= len(data):
                    block_align = struct.unpack("<H", data[pos + 8 + 12:pos + 8 + 14])[0]
                if cid == b"data":
                    if size in (0, 0xFFFFFFFF) or not block_align:   # streaming / unknown length: cannot tell
                        return None
                    present = len(data) - (pos + 8)
                    if present < size:
                        declared_s = size / block_align / sr
                        present_s = (present // block_align) / sr
                        return (f"This WAV file appears to be truncated: its header declares {declared_s:.2f} s of "
                                f"audio but only {present_s:.2f} s is present. Only the available audio was analysed.")
                    return None
                pos += 8 + size + (size & 1)
            return None
        if data[:4] == b"fLaC" and len(data) >= 42 and (data[4] & 0x7F) == 0:
            packed = struct.unpack(">Q", data[18:26])[0]
            flac_sr, total = packed >> 44, packed & 0xFFFFFFFFF
            if total and flac_sr and decoded_frames < total:
                return (f"This FLAC file appears to be truncated: its header declares {total / flac_sr:.2f} s of "
                        f"audio but only {decoded_frames / flac_sr:.2f} s could be decoded. Only the available "
                        "audio was analysed.")
    except struct.error:
        return None
    return None


def _read_bytes(src):
    try:
        if hasattr(src, "read"):
            data = src.read()
            if hasattr(src, "seek"):
                src.seek(0)
            return data
        return Path(src).read_bytes()
    except OSError as e:
        raise AudioError(f"The file could not be read ({type(e).__name__}).") from e


def inspect_audio(src):
    """Decode and validate a WAV/FLAC path or file-like object.

    Returns AudioCheck (mono float32 at 16 kHz + warnings). Raises AudioError for anything that must not be
    classified: undecodable or empty files, NaN/Inf samples, digital silence, constant (DC) signals,
    near-silent recordings, or less than MIN_SECONDS of audio. The decoded samples are exactly what the
    Phase 1 load_audio produced; only the checks are new.
    """
    data = _read_bytes(src)
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioError(
            f"The file could not be decoded as audio ({type(e).__name__}). "
            "Upload an uncompressed WAV or a FLAC file.") from e
    if not sr or sr <= 0:
        raise AudioError(f"The file reports an invalid sample rate ({sr}).")
    if x.shape[0] == 0:
        raise AudioError("The file contains no audio samples.")
    warnings = []
    trunc = detect_truncation(data, x.shape[0], sr)
    if trunc:
        warnings.append(trunc)
    x = x.mean(axis=1)                                   # stereo -> mono, as in training
    if not np.all(np.isfinite(x)):
        raise AudioError("The file contains invalid sample values (NaN or infinity).")
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()
    if not np.any(x):
        raise AudioError("The recording is completely silent (every sample is zero).")
    level = activity_level_dbfs(x)
    if level < MIN_ACTIVITY_DBFS:
        mean = float(np.mean(x))
        if abs(mean) >= DC_MEAN_MIN:
            raise AudioError(
                f"The recording contains only a constant (DC) signal with no audio variation (mean value {mean:.3f}). "
                "There is nothing for the model to analyse; this usually means a faulty recording or microphone.")
        raise AudioError(
            f"The recording is too quiet to analyse: its loudest parts are at {level:.0f} dBFS, below the "
            f"{MIN_ACTIVITY_DBFS:.0f} dBFS minimum for audible content. The model rescales every clip, so near-silent "
            "input would give a meaningless score. Check the microphone level or upload a recording with audible speech.")
    dur = len(x) / SR
    if dur < MIN_SECONDS:
        raise AudioError(
            f"The recording is only {dur:.2f} s long. At least {MIN_SECONDS:.0f} s of audio is needed; "
            "1 s is the shortest window the model was evaluated on. Please upload a longer recording "
            "(a few seconds of speech works best).")
    return AudioCheck(x.astype(np.float32, copy=False), int(sr), level, warnings)


def load_audio(src):
    """Phase 1 interface: returns (audio, original_sample_rate). Warnings are available via inspect_audio."""
    c = inspect_audio(src)
    return c.audio, c.orig_sr


def wrap_pad(x, n):
    """Repeat x from its start until it is n samples long (np.pad mode='wrap').

    This is the same padding vishing_experiments.load_wav applied to training
    and evaluation clips shorter than 4 s.
    """
    return x if len(x) >= n else np.pad(x, (0, n - len(x)), mode="wrap")


def make_windows(x, win_s, step_s):
    """Split audio into sliding windows of win_s seconds every step_s seconds.

    Audio shorter than one window is wrap-padded to exactly one window, the
    same way training handled short clips. Returns (windows [n, win*SR], end
    times in seconds of the ORIGINAL audio, padded flag).
    """
    W, S = int(round(win_s * SR)), int(round(step_s * SR))
    if W <= 0 or S <= 0:
        raise ValueError("Window and step must be positive.")
    dur = len(x) / SR
    padded = len(x) < W
    if padded:
        x = wrap_pad(x, W)
    starts = range(0, len(x) - W + 1, S)
    windows = np.stack([x[s:s + W] for s in starts]).astype(np.float32)
    ends = [min((s + W) / SR, dur) for s in starts]
    return windows, ends, padded


@torch.no_grad()
def score_windows(model, windows, batch=64):
    """CNN score per window: sigmoid(logit), in [0, 1].

    This is an UNCALIBRATED model score, not a probability that the call is a
    fraud. Higher means "more like the synthetic speech in ASVspoof 2019 LA".
    """
    out = []
    for i in range(0, len(windows), batch):
        xb = torch.from_numpy(windows[i:i + batch])
        out.append(torch.sigmoid(model(logmel(xb))).numpy())
    return np.concatenate(out).astype(float)


def ema(values, alpha):
    out, e = [], None
    for v in values:
        e = v if e is None else alpha * v + (1 - alpha) * e
        out.append(e)
    return out


def heuristic_band(score):
    lo, hi = HEURISTIC_BANDS
    return "LOW" if score < lo else ("MEDIUM" if score < hi else "HIGH")
