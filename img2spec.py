#!/usr/bin/env python3
"""img2spec - render an image as a spectrogram and synthesize a WAV from it.

Give this script a picture and it produces a mono ``.wav`` whose short-time
Fourier transform magnitude looks like that picture: image columns become time,
image rows become frequency, brightness becomes loudness.

The catch (and it is a mathematical one, not a bug): an STFT is complex, the
image only supplies the magnitude, and the phase is unknown.  Reconstruction is
therefore an optimisation problem, solved here with (fast) Griffin-Lim
iteration.  The result is recognisably the image, with a characteristic
"phasey" timbre.  See README.md for what can and cannot be expected.

Examples
--------
    python img2spec.py logo.png logo.wav
    python img2spec.py photo.jpg out.wav --duration 8 --freq-scale log
    python img2spec.py text.png out.wav --verify out_spectrogram.png
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

try:  # scipy's FFT is faster; numpy is the fallback so the script never hard-fails
    from scipy.fft import irfft, rfft
except ImportError:  # pragma: no cover
    from numpy.fft import irfft, rfft

TINY = 1e-12
CHANNELS = ("luma", "mean", "r", "g", "b")
FREQ_SCALES = ("linear", "log", "mel")
BITS = (16, 24, 32, "float")


# --------------------------------------------------------------------------- #
# image -> magnitude spectrogram
# --------------------------------------------------------------------------- #

def to_gray(img: Image.Image, channel: str = "luma") -> np.ndarray:
    """Flatten *img* to a 2-D float array in [0, 255].  Row 0 is the TOP row."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")

    if img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        img = Image.alpha_composite(canvas, rgba)

    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    if channel == "r":
        return rgb[..., 0]
    if channel == "g":
        return rgb[..., 1]
    if channel == "b":
        return rgb[..., 2]
    if channel == "mean":
        return rgb.mean(axis=2)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def row_frequencies(n_rows: int, scale: str, f_min: float, f_max: float) -> np.ndarray:
    """Frequency in Hz that each image row stands for; row 0 is the lowest."""
    if n_rows < 1:
        raise ValueError("need at least one row")
    t = np.linspace(0.0, 1.0, n_rows)
    if scale == "linear":
        return f_min + t * (f_max - f_min)
    if scale == "log":
        if f_min <= 0.0:
            raise ValueError("--fmin must be > 0 for a log frequency scale")
        return f_min * (f_max / f_min) ** t
    if scale == "mel":
        # mel(0) is 0, so a mel axis is happy to start at DC
        return _mel_to_hz(
            _hz_to_mel(f_min) + t * (_hz_to_mel(f_max) - _hz_to_mel(f_min))
        )
    raise ValueError(f"unknown frequency scale {scale!r}; expected {FREQ_SCALES}")


def remap_rows(mag: np.ndarray, src_freqs: np.ndarray, dst_freqs: np.ndarray) -> np.ndarray:
    """Linearly interpolate the rows of *mag* from *src_freqs* onto *dst_freqs*.

    Frequencies outside the source range clamp to the first/last row, which is
    what we want when the image does not cover the whole spectrum.
    """
    n_src = mag.shape[0]
    rows = np.interp(dst_freqs, src_freqs, np.arange(n_src, dtype=np.float64))
    lo = np.clip(np.floor(rows).astype(np.int64), 0, n_src - 1)
    hi = np.clip(lo + 1, 0, n_src - 1)
    w = (rows - lo)[:, None]
    return mag[lo] * (1.0 - w) + mag[hi] * w


def image_to_magnitude(
    img: Image.Image,
    *,
    n_frames: int,
    n_fft: int,
    sample_rate: int,
    channel: str = "luma",
    freq_scale: str = "log",
    f_min: float = 32.0,
    f_max: float | None = None,
    min_db: float = -80.0,
    max_db: float = 0.0,
    gamma: float = 1.0,
    invert_y: bool = True,
    zero_black: bool = True,
) -> np.ndarray:
    """Convert an image into an ``(n_bins, n_frames)`` linear-magnitude target.

    Pixel brightness is mapped to decibels (not to amplitude directly - that
    would just sound like white noise), then to linear magnitude.
    """
    n_bins = n_fft // 2 + 1
    nyquist = sample_rate / 2.0
    f_max = nyquist if (f_max is None or f_max <= 0.0) else min(float(f_max), nyquist)

    gray = to_gray(img, channel)
    if invert_y:
        gray = np.flipud(gray)  # image row 0 is the top = the treble end
    if gamma != 1.0 and gamma > 0.0:
        gray = 255.0 * np.power(np.clip(gray, 0.0, 255.0) / 255.0, gamma)

    # PIL resizes to (width, height) = (time, frequency).
    src = Image.fromarray(np.ascontiguousarray(gray, dtype=np.float32), mode="F")
    disp = np.asarray(src.resize((n_frames, n_bins), Image.BICUBIC), dtype=np.float64)
    disp = np.clip(disp, 0.0, 255.0)

    db = min_db + (disp / 255.0) * (max_db - min_db)
    mag = np.power(10.0, db / 20.0)
    if zero_black:
        # Bicubic undershoot clips to 0; treat those pixels as true silence
        # rather than as the -80 dB noise floor.
        mag[disp <= 0.0] = 0.0

    if freq_scale != "linear":
        lin_freqs = np.arange(n_bins, dtype=np.float64) * sample_rate / n_fft
        disp_freqs = row_frequencies(n_bins, freq_scale, f_min, f_max)
        mag = remap_rows(mag, disp_freqs, lin_freqs)
        mag[lin_freqs < f_min] = 0.0  # image content does not reach below f_min

    mag[0] = 0.0  # never synthesize DC: it is inaudible and eats headroom
    return mag


# --------------------------------------------------------------------------- #
# STFT / ISTFT / Griffin-Lim
# --------------------------------------------------------------------------- #

def hann_window(n_fft: int) -> np.ndarray:
    """Periodic Hann window (the one that actually satisfies COLA)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)


def _frames(y: np.ndarray, n_fft: int, hop: int, n_frames: int) -> np.ndarray:
    starts = np.arange(n_frames, dtype=np.int64) * hop
    return y[starts[:, None] + np.arange(n_fft, dtype=np.int64)[None, :]]


def stft(y: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Complex STFT, shape ``(n_bins, n_frames)``."""
    n_frames = 1 + (len(y) - n_fft) // hop
    return rfft(_frames(y, n_fft, hop, n_frames) * window, axis=-1).T


def stft_mag(y: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    n_frames = 1 + (len(y) - n_fft) // hop
    return np.abs(rfft(_frames(y, n_fft, hop, n_frames) * window, axis=-1)).T


def istft(
    spec: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    length: int,
    normalize: str = "scalar",
) -> np.ndarray:
    """Weighted overlap-add inverse STFT of a ``(n_bins, n_frames)`` spectrum.

    Normalisation is by a single scalar - the steady-state window sum - rather
    than sample by sample.  Dividing by the per-sample sum looks more principled
    and does reconstruct the interior to machine precision, but the first and
    last ``n_fft`` samples have a window sum near zero, and dividing by it
    amplifies whatever the phase estimate got wrong there by ~5 orders of
    magnitude.  The result is a huge click at each end that then dominates peak
    normalisation and squashes the entire body of the audio to the noise floor.
    With a scalar the extremes simply fade in and out over ``n_fft`` samples,
    which is what a real STFT front end does anyway.  ``normalize='exact'``
    restores the per-sample division (used by the round-trip test).
    """
    frames = irfft(spec.T, n=n_fft, axis=-1) * window
    y = np.zeros(length, dtype=np.float64)
    wsum = np.zeros(length, dtype=np.float64)
    w2 = window * window
    for i in range(spec.shape[1]):
        s = i * hop
        y[s : s + n_fft] += frames[i]
        wsum[s : s + n_fft] += w2
    if normalize == "exact":
        return y / np.maximum(wsum, 1e-9)
    if normalize != "scalar":
        raise ValueError(f"unknown istft normalization {normalize!r}")
    return y / max(float(wsum.max()), TINY)


def griffin_lim(
    mag: np.ndarray,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    n_iter: int = 64,
    momentum: float = 0.99,
    seed: int = 0,
    window: np.ndarray | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a signal whose STFT magnitude is *mag*, via fast Griffin-Lim.

    Returns ``(signal, window)``.  ``n_iter <= 0`` degenerates to plain inverse
    STFT with random phase, which is the noisy baseline worth listening to once.
    """
    mag = np.asarray(mag, dtype=np.float64)
    n_bins, n_frames = mag.shape
    if n_bins != n_fft // 2 + 1:
        raise ValueError(f"magnitude has {n_bins} bins, expected {n_fft // 2 + 1}")
    if window is None:
        window = hann_window(n_fft)

    length = n_fft + hop * (n_frames - 1)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, size=mag.shape)

    if n_iter <= 0:
        return istft(mag * np.exp(1j * phi), n_fft, hop, window, length), window

    phi_hat = phi
    report_every = max(1, n_iter // 10)
    for i in range(n_iter):
        y = istft(mag * np.exp(1j * phi), n_fft, hop, window, length)
        phi_hat = np.angle(stft(y, n_fft, hop, window))
        phi = phi_hat + momentum * (phi_hat - phi)  # Perraudin et al., fast GL
        if progress and (i + 1) % report_every == 0:
            print(f"  griffin-lim {i + 1}/{n_iter}", file=sys.stderr)

    # Resynthesise from the phase we actually measured, so the output is
    # self-consistent rather than momentum-extrapolated.
    y = istft(mag * np.exp(1j * phi_hat), n_fft, hop, window, length)
    return y, window


# --------------------------------------------------------------------------- #
# loudness / IO
# --------------------------------------------------------------------------- #

def normalize(y: np.ndarray, mode: str = "peak", target_db: float = -1.0) -> np.ndarray:
    """Scale *y* so it does not clip.  ``mode='none'`` is a no-op."""
    y = np.asarray(y, dtype=np.float64)
    if mode == "none" or y.size == 0:
        return y
    if mode == "peak":
        ref = float(np.max(np.abs(y)))
    elif mode == "rms":
        ref = float(np.sqrt(np.mean(y * y)))
    else:
        raise ValueError(f"unknown normalize mode {mode!r}; expected peak|rms|none")
    if ref <= TINY:
        return y
    return y * (10.0 ** (target_db / 20.0) / ref)


def write_wav(path: str | Path, y: np.ndarray, sample_rate: int = 22050, bits=16) -> None:
    """Write mono PCM (or float) WAV.  Falls back to the stdlib for 16-bit."""
    if bits not in BITS:
        raise ValueError(f"--bits must be one of {BITS}")
    y = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    if sf is not None:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32", "float": "FLOAT"}[bits]
        sf.write(str(path), y, sample_rate, subtype=subtype)
        return

    if bits != 16:
        raise RuntimeError(
            f"writing {bits}-bit WAV needs the optional 'soundfile' package; "
            "use --bits 16 or `pip install soundfile`"
        )
    pcm = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def spectral_convergence(target: np.ndarray, actual: np.ndarray) -> float:
    """||target - actual|| / ||target|| after matching both peaks.  0 is perfect."""
    a = target / max(float(target.max()), TINY)
    b = actual / max(float(actual.max()), TINY)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), TINY))


def db_correlation(target: np.ndarray, actual: np.ndarray, floor_db: float = -100.0) -> float:
    """Pearson correlation of the two spectrograms in dB.  1.0 is perfect."""
    floor = 10.0 ** (floor_db / 20.0)
    a = 20.0 * np.log10(np.maximum(target / max(float(target.max()), TINY), floor))
    b = 20.0 * np.log10(np.maximum(actual / max(float(actual.max()), TINY), floor))
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def save_spectrogram_png(
    mag: np.ndarray, path: str | Path, min_db: float = -80.0, max_db: float = 0.0
) -> None:
    """Render a magnitude spectrogram as an 8-bit grayscale PNG."""
    db = 20.0 * np.log10(np.maximum(mag, TINY))
    v = np.clip((db - min_db) / max(max_db - min_db, TINY), 0.0, 1.0)
    Image.fromarray(np.round(v * 255.0).astype(np.uint8), mode="L").save(str(path))


def save_comparison_png(
    target: np.ndarray,
    actual: np.ndarray,
    path: str | Path,
    min_db: float = -80.0,
    max_db: float = 0.0,
) -> None:
    """Write `target | actual` side by side so the reconstruction can be eyeballed.

    Both panels live on the linear frequency grid the audio actually uses, so
    under --freq-scale log they look frequency-warped compared with the source
    image.  The point is that the two panels match each other.
    """
    panels = []
    for m in (target, actual):
        db = 20.0 * np.log10(np.maximum(m, TINY))
        v = np.clip((db - min_db) / max(max_db - min_db, TINY), 0.0, 1.0)
        panels.append(np.round(v * 255.0).astype(np.uint8))
    sep = np.full((panels[0].shape[0], 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([panels[0], sep, panels[1]], axis=1), mode="L").save(
        str(path)
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def plan_timing(
    image_size: tuple[int, int],
    *,
    frames: int | None = None,
    hop: int = 512,
    duration: float | None = None,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    max_duration: float = 60.0,
) -> tuple[int, int, float, list[str]]:
    """Decide how many frames to use and what the effective hop is.

    ``--frames`` wins; otherwise one image column becomes one STFT frame.
    Resampling time to preserve the image's pixel aspect ratio instead is
    tempting but wrong: a spectrogram's axes are Hz and seconds, so its aspect
    ratio carries no meaning, and matching it would resample time by
    ``n_bins / height`` (3.2x for a 320 px tall image) for nothing.
    ``--duration`` then overrides ``--hop`` so the result lasts that long, and
    finally the frame count is clamped to ``--max-duration``.
    """
    width = image_size[0]
    if frames is not None:
        n_frames = int(frames)
    else:
        n_frames = int(width)
    n_frames = max(1, n_frames)

    if duration is not None and n_frames > 1:
        hop_eff = int(round((duration * sample_rate - n_fft) / (n_frames - 1)))
    else:
        hop_eff = int(hop)
    hop_eff = max(1, hop_eff)

    notes: list[str] = []
    if max_duration and max_duration > 0.0 and n_frames > 1:
        limit = int((max_duration * sample_rate - n_fft) / hop_eff) + 1
        if n_frames > limit >= 1:
            implied = (n_fft + hop_eff * (n_frames - 1)) / sample_rate
            notes.append(
                f"image implies {n_frames} frames ({implied:.1f}s); clamped to "
                f"--max-duration {max_duration:g}s ({limit} frames). "
                f"Pass --max-duration 0 to disable."
            )
            n_frames = limit

    if hop_eff < n_fft // 16:
        notes.append(
            f"hop {hop_eff} is {n_fft / hop_eff:.0f}x overlap: slow, and far more "
            f"redundant than the phase estimate can exploit. Raise --duration, "
            f"or lower --frames (currently {n_frames})."
        )

    length = n_fft + hop_eff * (n_frames - 1)
    return n_frames, hop_eff, length / sample_rate, notes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="img2spec",
        description="Synthesize a .wav whose STFT magnitude spectrogram is an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Reconstruction is Griffin-Lim: the phase is estimated, so the "
        "result is recognisably the image but not a bit-exact inversion.",
    )
    p.add_argument("image", help="source image (any format Pillow can read)")
    p.add_argument("output", help="destination .wav file")

    g = p.add_argument_group("timing")
    g.add_argument("--sr", type=int, default=22050, help="sample rate in Hz")
    g.add_argument("--n-fft", type=int, default=2048, help="FFT size; sets the number of frequency bins")
    g.add_argument("--hop", type=int, default=512, help="samples between frames (smaller = longer, smoother)")
    g.add_argument("--frames", type=int, default=None, help="force the number of STFT frames")
    g.add_argument("--duration", type=float, default=None, help="force the output length in seconds (overrides --hop)")
    g.add_argument("--max-duration", type=float, default=60.0, help="clamp long outputs; 0 disables")

    g = p.add_argument_group("image mapping")
    g.add_argument("--channel", choices=CHANNELS, default="luma", help="which image channel to sonify")
    g.add_argument("--freq-scale", choices=FREQ_SCALES, default="log", help="how image rows map to frequency")
    g.add_argument("--fmin", type=float, default=32.0, help="lowest frequency for log/mel scaling, in Hz")
    g.add_argument("--fmax", type=float, default=None, help="highest frequency, in Hz (default: Nyquist)")
    g.add_argument("--gamma", type=float, default=1.0, help="pre-warp brightness: >1 darkens, <1 brightens")
    g.add_argument("--no-invert-y", action="store_true", help="do not flip the image vertically")
    g.add_argument("--no-zero-black", action="store_true", help="treat black as the noise floor, not silence")

    g = p.add_argument_group("dynamics")
    g.add_argument("--min-db", type=float, default=-80.0, help="dB level that black maps to (the timbre knob)")
    g.add_argument("--max-db", type=float, default=0.0, help="dB level that white maps to")
    g.add_argument("--normalize", choices=("peak", "rms", "none"), default="peak", help="output gain policy")
    g.add_argument("--target-db", type=float, default=-1.0, help="target level for --normalize")
    g.add_argument("--bits", choices=BITS, default=16, help="output bit depth")

    g = p.add_argument_group("reconstruction")
    g.add_argument("--iterations", type=int, default=64, help="Griffin-Lim iterations; 0 = random phase")
    g.add_argument("--momentum", type=float, default=0.99, help="fast Griffin-Lim momentum")
    g.add_argument("--seed", type=int, default=0, help="random seed for the initial phase")

    g = p.add_argument_group("output")
    g.add_argument("--verify", metavar="PNG", default=None, help="write a target|result spectrogram comparison")
    g.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    warn = lambda *a: print("warning:", *a, file=sys.stderr)

    if args.n_fft < 8 or args.n_fft & (args.n_fft - 1):
        print("error: --n-fft must be a power of two >= 8", file=sys.stderr)
        return 2

    try:
        img = Image.open(args.image)
    except OSError as exc:
        print(f"error: cannot read image {args.image!r}: {exc}", file=sys.stderr)
        return 1

    n_bins = args.n_fft // 2 + 1
    with img:
        n_frames, hop, duration, notes = plan_timing(
            img.size,
            frames=args.frames,
            hop=args.hop,
            duration=args.duration,
            sample_rate=args.sr,
            n_fft=args.n_fft,
            max_duration=args.max_duration,
        )
        for note in notes:
            warn(note)
        say(
            f"image {img.size[0]}x{img.size[1]} -> {n_frames} frames x {n_bins} bins, "
            f"hop {hop}, {duration:.2f}s @ {args.sr} Hz"
        )
        mag = image_to_magnitude(
            img,
            n_frames=n_frames,
            n_fft=args.n_fft,
            sample_rate=args.sr,
            channel=args.channel,
            freq_scale=args.freq_scale,
            f_min=args.fmin,
            f_max=args.fmax,
            min_db=args.min_db,
            max_db=args.max_db,
            gamma=args.gamma,
            invert_y=not args.no_invert_y,
            zero_black=not args.no_zero_black,
        )

    if not np.any(mag > 0.0):
        warn("the target spectrogram is all zeros; writing silence")

    say(f"griffin-lim: {args.iterations} iterations (momentum {args.momentum})")
    y, window = griffin_lim(
        mag,
        n_fft=args.n_fft,
        hop=hop,
        n_iter=args.iterations,
        momentum=args.momentum,
        seed=args.seed,
        progress=not args.quiet,
    )

    y = normalize(y, args.normalize, args.target_db)
    write_wav(args.output, y, args.sr, args.bits)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    print(
        f"wrote {args.output}: {len(y) / args.sr:.2f}s, peak {peak:.4f}, "
        f"rms {20.0 * np.log10(max(rms, TINY)):.1f} dBFS"
    )

    if args.verify:
        actual = stft_mag(y, args.n_fft, hop, window)
        if actual.shape != mag.shape:
            actual = actual[:, : mag.shape[1]]
        # The outermost frames are the STFT's own fade-in/fade-out, not a
        # reconstruction failure, so they are excluded from the scores.
        edge = int(np.ceil(args.n_fft / hop))
        if actual.shape[1] > 2 * edge + 2:
            inner = slice(edge, actual.shape[1] - edge)
            cmp_target, cmp_actual = mag[:, inner], actual[:, inner]
        else:
            cmp_target, cmp_actual = mag, actual
        sc = spectral_convergence(cmp_target, cmp_actual)
        corr = db_correlation(cmp_target, cmp_actual)
        save_comparison_png(mag, actual, args.verify, args.min_db, args.max_db)
        print(
            f"verify: spectral convergence {sc:.4f} (lower is better), "
            f"dB correlation {corr:.4f} (higher is better); "
            f"inner {cmp_target.shape[1]} frames -> {args.verify}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
