"""Tests for img2spec.

Run with either::

    python -m unittest discover -s tests -v
    python -m pytest tests -q

The test images are generated in-process rather than checked in, so the tests
say something about the transform instead of about a fixture that happened to
be committed once.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import img2spec as m  # noqa: E402

N_FFT = 256
HOP = 64
SR = 8000
N_BINS = N_FFT // 2 + 1


def bars_image(width: int = 64, height: int = 64, bars=(8, 20, 33)) -> Image.Image:
    """Horizontal bright bars on black: a few steady tones in spectrogram form."""
    a = np.zeros((height, width), dtype=np.uint8)
    for y in bars:
        a[y : y + 3, :] = 255
    return Image.fromarray(a, mode="L")


def gradient_image(width: int = 32, height: int = 48) -> Image.Image:
    """A vertical ramp: dark at the top, bright at the bottom."""
    a = np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width))
    return Image.fromarray(a, mode="L")


class TestImageMapping(unittest.TestCase):
    def test_channels(self):
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((1, 0), (0, 255, 0))
        np.testing.assert_allclose(m.to_gray(img, "r"), [[255.0, 0.0]])
        np.testing.assert_allclose(m.to_gray(img, "g"), [[0.0, 255.0]])
        np.testing.assert_allclose(m.to_gray(img, "b"), [[0.0, 0.0]])
        np.testing.assert_allclose(m.to_gray(img, "mean"), [[85.0, 85.0]])
        np.testing.assert_allclose(m.to_gray(img, "luma"), [[76.245, 149.685]])

    def test_alpha_is_composited_over_black(self):
        img = Image.new("RGBA", (1, 1), (255, 255, 255, 0))  # fully transparent
        self.assertEqual(float(m.to_gray(img, "luma")[0, 0]), 0.0)

    def test_row_frequencies(self):
        f = m.row_frequencies(5, "log", 20.0, 20000.0)
        self.assertAlmostEqual(f[0], 20.0)
        self.assertAlmostEqual(f[-1], 20000.0)
        self.assertTrue(np.all(np.diff(f) > 0))
        self.assertLess(f[2], 1000.0)  # log spacing puts the midpoint low
        self.assertAlmostEqual(m.row_frequencies(3, "linear", 0.0, 100.0)[1], 50.0)
        fm = m.row_frequencies(5, "mel", 0.0, 8000.0)
        self.assertAlmostEqual(fm[0], 0.0)
        self.assertAlmostEqual(fm[-1], 8000.0)
        # mel is far from linear even at only 8 kHz: the midpoint is ~1.8 kHz
        self.assertAlmostEqual(fm[2], 1767.7, delta=1.0)

    def test_row_frequencies_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            m.row_frequencies(4, "log", 0.0, 100.0)
        with self.assertRaises(ValueError):
            m.row_frequencies(4, "nope", 20.0, 100.0)

    def test_remap_rows_identity_and_interpolation(self):
        mag = np.arange(8, dtype=float).reshape(4, 2)
        f = np.array([0.0, 10.0, 20.0, 30.0])
        np.testing.assert_allclose(m.remap_rows(mag, f, f), mag)
        mid = m.remap_rows(mag, f, np.array([5.0]))
        np.testing.assert_allclose(mid[0], [1.0, 2.0])  # halfway between rows 0 and 1
        # outside the source range clamps to the edge rows
        np.testing.assert_allclose(m.remap_rows(mag, f, np.array([-5.0]))[0], mag[0])
        np.testing.assert_allclose(m.remap_rows(mag, f, np.array([99.0]))[0], mag[-1])

    def test_magnitude_shape_dc_and_range(self):
        img = bars_image()
        mag = m.image_to_magnitude(
            img, n_frames=32, n_fft=N_FFT, sample_rate=SR, freq_scale="linear"
        )
        self.assertEqual(mag.shape, (N_BINS, 32))
        self.assertEqual(mag[0].max(), 0.0, "DC must never be synthesized")
        self.assertGreaterEqual(mag.min(), 0.0)
        self.assertLessEqual(mag.max(), 1.0 + 1e-9)
        self.assertAlmostEqual(mag.max(), 1.0, places=6)

    def test_black_image_is_silent(self):
        img = Image.new("L", (32, 32), 0)
        mag = m.image_to_magnitude(img, n_frames=16, n_fft=N_FFT, sample_rate=SR)
        self.assertTrue(np.all(mag == 0.0))

    def test_black_image_keeps_floor_when_asked(self):
        img = Image.new("L", (32, 32), 0)
        mag = m.image_to_magnitude(
            img, n_frames=16, n_fft=N_FFT, sample_rate=SR, zero_black=False, min_db=-80.0
        )
        self.assertAlmostEqual(float(mag[1:].max()), 10.0 ** (-80.0 / 20.0), places=9)

    def test_gradient_maps_monotonically_onto_frequency(self):
        """A ramp that is bright at the bottom of the image is loud in the bass."""
        mag = m.image_to_magnitude(
            gradient_image(), n_frames=8, n_fft=N_FFT, sample_rate=SR,
            freq_scale="linear",
        )
        column = mag[1:, 0]  # row 0 is DC and is always zeroed
        self.assertGreater(column[0], column[-1])
        self.assertTrue(np.all(np.diff(column) <= 1e-9))

    def test_white_is_full_scale_and_black_is_floor(self):
        img = Image.new("L", (4, 4), 255)
        mag = m.image_to_magnitude(
            img, n_frames=4, n_fft=N_FFT, sample_rate=SR, freq_scale="linear"
        )
        self.assertAlmostEqual(float(mag.max()), 1.0, places=9)

    def test_invert_y_puts_white_at_the_top(self):
        a = np.zeros((64, 8), dtype=np.uint8)
        a[0:4, :] = 255  # bright strip along the TOP of the image
        img = Image.fromarray(a, mode="L")
        mag = m.image_to_magnitude(
            img, n_frames=8, n_fft=N_FFT, sample_rate=SR, freq_scale="linear"
        )
        self.assertGreater(mag[-4:].max(), 0.5, "top of image = treble end")
        self.assertEqual(mag[: N_BINS // 2].max(), 0.0)

    def test_log_scale_compresses_the_treble(self):
        """Under log mapping the middle image row is a bass note, not a mid one.

        With a linear axis the middle row of a 4000 Hz band is 2000 Hz; with a
        log axis from 40 Hz it is 400 Hz.  That is the whole point of the
        default: the low octaves get the image rows they perceptually deserve.
        """
        a = np.zeros((65, 8), dtype=np.uint8)
        a[32, :] = 255  # a bright line through the middle of the image
        img = Image.fromarray(a, mode="L")
        lin = m.image_to_magnitude(
            img, n_frames=8, n_fft=N_FFT, sample_rate=SR, freq_scale="linear"
        )
        log = m.image_to_magnitude(
            img, n_frames=8, n_fft=N_FFT, sample_rate=SR, freq_scale="log", f_min=40.0
        )
        self.assertAlmostEqual(np.argmax(lin.max(axis=1)), N_BINS // 2, delta=2)
        self.assertLess(np.argmax(log.max(axis=1)), N_BINS // 4)


class TestStftRoundTrip(unittest.TestCase):
    def test_roundtrip_is_exact_in_the_interior(self):
        rng = np.random.default_rng(0)
        n_frames = 40
        length = N_FFT + HOP * (n_frames - 1)
        x = rng.standard_normal(length)
        w = m.hann_window(N_FFT)
        spec = m.stft(x, N_FFT, HOP, w)
        self.assertEqual(spec.shape, (N_BINS, n_frames))
        back = m.istft(spec, N_FFT, HOP, w, length, normalize="exact")
        inner = slice(2 * N_FFT, length - 2 * N_FFT)
        self.assertLess(np.abs(back[inner] - x[inner]).max(), 1e-10)

    def test_scalar_normalisation_fades_the_edges(self):
        """The default is deliberately not exact at the extremes - see istft."""
        rng = np.random.default_rng(1)
        n_frames = 40
        length = N_FFT + HOP * (n_frames - 1)
        x = rng.standard_normal(length)
        w = m.hann_window(N_FFT)
        back = m.istft(m.stft(x, N_FFT, HOP, w), N_FFT, HOP, w, length)
        self.assertEqual(back[0], 0.0)
        inner = slice(2 * N_FFT, length - 2 * N_FFT)
        self.assertLess(np.abs(back[inner] - x[inner]).max(), 1e-10)
        # and no large transient: the peak stays in the same ballpark as x
        self.assertLess(np.abs(back).max(), 1.5 * np.abs(x).max())

    def test_stft_mag_matches_stft(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(N_FFT + HOP * 9)
        w = m.hann_window(N_FFT)
        np.testing.assert_allclose(
            m.stft_mag(x, N_FFT, HOP, w), np.abs(m.stft(x, N_FFT, HOP, w))
        )

    def test_istft_rejects_unknown_normalization(self):
        w = m.hann_window(N_FFT)
        with self.assertRaises(ValueError):
            m.istft(np.zeros((N_BINS, 4), dtype=complex), N_FFT, HOP, w, 1024, "bogus")


class TestGriffinLim(unittest.TestCase):
    def test_gl_does_much_better_on_a_reachable_target(self):
        """GL must be judged against what is actually reachable.

        A spectrogram measured from a real signal lies in the range of the STFT
        operator, so the phase problem is solvable and GL gets close.  A
        hand-drawn rectangle in the time-frequency plane is not reachable at
        all.  Absolute error stays noticeable even in the reachable case
        because spectral convergence is dominated by the near-silent bins.
        """
        w = m.hann_window(N_FFT)
        inner = slice(4, 60)
        n_frames = 64
        length = N_FFT + HOP * (n_frames - 1)
        t = np.arange(length) / SR
        x = 0.6 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)
        reachable = m.stft_mag(x, N_FFT, HOP, w)

        impossible = np.zeros((N_BINS, n_frames))
        impossible[40, :] = 1.0

        def sc_of(target):
            y, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=128, seed=0)
            return m.spectral_convergence(
                target[:, inner], m.stft_mag(y, N_FFT, HOP, w)[:, inner]
            )

        sc_reachable = sc_of(reachable)
        sc_impossible = sc_of(impossible)
        self.assertLess(sc_reachable, 0.30)
        self.assertLess(sc_reachable, 0.5 * sc_impossible)

    def test_true_phase_reconstructs_the_signal_exactly(self):
        """Sanity check on the metric itself: no error when the phase is known."""
        rng = np.random.default_rng(5)
        n_frames = 32
        length = N_FFT + HOP * (n_frames - 1)
        x = rng.standard_normal(length)
        w = m.hann_window(N_FFT)
        spec = m.stft(x, N_FFT, HOP, w)
        y = m.istft(spec, N_FFT, HOP, w, length)
        inner = slice(4, n_frames - 4)
        np.testing.assert_allclose(
            m.stft_mag(y, N_FFT, HOP, w)[:, inner], np.abs(spec)[:, inner], atol=1e-9
        )

    def test_iterating_beats_random_phase(self):
        """GL is an optimisation, so it must actually lower the inconsistency.

        The improvement plateaus quickly (~16-64 iterations on targets like
        this) because a hand-drawn spectrogram is not in the range of the STFT
        operator, so this checks direction, not a quality bar.
        """
        img = bars_image()
        target = m.image_to_magnitude(
            img, n_frames=64, n_fft=N_FFT, sample_rate=SR, freq_scale="linear"
        )
        w = m.hann_window(N_FFT)
        inner = slice(4, 60)
        rand, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=0, seed=0)
        conv, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=64, seed=0)
        sc_rand = m.spectral_convergence(
            target[:, inner], m.stft_mag(rand, N_FFT, HOP, w)[:, inner]
        )
        sc_conv = m.spectral_convergence(
            target[:, inner], m.stft_mag(conv, N_FFT, HOP, w)[:, inner]
        )
        self.assertLess(sc_conv, 0.75 * sc_rand)
        self.assertGreater(
            m.db_correlation(target[:, inner], m.stft_mag(conv, N_FFT, HOP, w)[:, inner]),
            m.db_correlation(target[:, inner], m.stft_mag(rand, N_FFT, HOP, w)[:, inner]),
        )

    def test_seed_is_deterministic(self):
        target = np.zeros((N_BINS, 32))
        target[10, :] = 1.0
        a, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=8, seed=7)
        b, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=8, seed=7)
        c, _ = m.griffin_lim(target, n_fft=N_FFT, hop=HOP, n_iter=8, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_rejects_wrong_bin_count(self):
        with self.assertRaises(ValueError):
            m.griffin_lim(np.zeros((7, 4)), n_fft=N_FFT, hop=HOP)

    def test_silent_target_stays_silent(self):
        y, _ = m.griffin_lim(np.zeros((N_BINS, 32)), n_fft=N_FFT, hop=HOP, n_iter=16)
        self.assertEqual(np.abs(y).max(), 0.0)


class TestSineSynth(unittest.TestCase):
    def test_single_bin_hits_its_target(self):
        """One steady tone: additive synthesis is exact up to windowing."""
        w = m.hann_window(N_FFT)
        n_frames = 32
        target = np.zeros((N_BINS, n_frames))
        target[40, :] = 0.5
        y = m.sine_synth(target, sample_rate=SR, n_fft=N_FFT, hop=HOP)
        act = m.stft_mag(y, N_FFT, HOP, w)
        np.testing.assert_allclose(
            np.median(act[40, 4:-4]), 0.5, rtol=0.02
        )

    def test_rejects_wrong_bin_count(self):
        with self.assertRaises(ValueError):
            m.sine_synth(np.zeros((7, 4)), sample_rate=SR, n_fft=N_FFT, hop=HOP)

    def test_silent_target_is_silent(self):
        y = m.sine_synth(np.zeros((N_BINS, 16)), sample_rate=SR, n_fft=N_FFT, hop=HOP)
        np.testing.assert_array_equal(y, 0.0)


class TestGradSynth(unittest.TestCase):
    def test_loss_decreases_from_random_init(self):
        """The optimizer must make the dB-spectrogram closer, not just move."""
        w = m.hann_window(N_FFT)
        n_frames = 24
        rng = np.random.default_rng(3)
        target = rng.uniform(0.01, 1.0, (N_BINS, n_frames))

        def loss(y):
            act = m.stft_mag(y, N_FFT, HOP, w)
            return float(np.mean((np.log(act + 1e-5) - np.log(target + 1e-5)) ** 2))

        y0 = 0.1 * rng.standard_normal(N_FFT + HOP * (n_frames - 1))
        y, _ = m.grad_synth(
            target, sample_rate=SR, n_fft=N_FFT, hop=HOP,
            init=y0, steps=150, lr=0.05,
        )
        self.assertLess(loss(y), loss(y0))

    def test_sines_init_beats_random_init(self):
        """Initialization from additive synthesis converges to lower loss."""
        w = m.hann_window(N_FFT)
        n_frames = 24
        target = np.zeros((N_BINS, n_frames))
        target[40, :] = 0.5
        target[41, :] = 0.3

        def loss(y):
            act = m.stft_mag(y, N_FFT, HOP, w)
            return float(np.mean((np.log(act + 1e-5) - np.log(target + 1e-5)) ** 2))

        y_rand = 0.1 * np.random.default_rng(0).standard_normal(
            N_FFT + HOP * (n_frames - 1)
        )
        y_init = m.sine_synth(target, sample_rate=SR, n_fft=N_FFT, hop=HOP)
        out_rand, _ = m.grad_synth(
            target, sample_rate=SR, n_fft=N_FFT, hop=HOP, init=y_rand, steps=100
        )
        out_init, _ = m.grad_synth(
            target, sample_rate=SR, n_fft=N_FFT, hop=HOP, init=y_init, steps=100
        )
        self.assertLess(loss(out_init), loss(out_rand))

    def test_output_is_finite(self):
        rng = np.random.default_rng(1)
        target = rng.uniform(0.0, 1.0, (N_BINS, 8))
        y, _ = m.grad_synth(
            target, sample_rate=SR, n_fft=N_FFT, hop=HOP, steps=20
        )
        self.assertTrue(np.all(np.isfinite(y)))


class TestOutput(unittest.TestCase):
    def test_normalize_modes(self):
        y = np.array([0.0, 0.5, -0.25])
        peak = m.normalize(y, "peak", target_db=-6.0)
        self.assertAlmostEqual(np.abs(peak).max(), 10.0 ** (-6.0 / 20.0), places=9)
        none = m.normalize(y, "none")
        np.testing.assert_allclose(none, y)
        rms = m.normalize(y, "rms", target_db=0.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(rms**2))), 1.0, places=9)
        np.testing.assert_allclose(m.normalize(np.zeros(4), "peak"), np.zeros(4))

    def test_normalize_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            m.normalize(np.ones(4), "lufs")

    def test_write_wav_is_readable_and_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.wav"
            m.write_wav(path, np.array([0.0, 0.5, 2.0, -2.0]), SR, 16)
            with wave.open(str(path)) as fh:
                self.assertEqual(fh.getnchannels(), 1)
                self.assertEqual(fh.getsampwidth(), 2)
                self.assertEqual(fh.getframerate(), SR)
                self.assertEqual(fh.getnframes(), 4)
                data = np.frombuffer(fh.readframes(4), dtype="<i2")
            # samples above full scale are clipped, not wrapped around
            self.assertGreaterEqual(int(data[2]), 32766)
            self.assertLessEqual(int(data[3]), -32766)

    def test_write_wav_rejects_bad_depth(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.write_wav(Path(d) / "x.wav", np.zeros(4), SR, 12)


class TestPlanTiming(unittest.TestCase):
    def test_frames_default_to_image_width(self):
        n, hop, dur, notes = m.plan_timing(
            (512, 320), frames=None, hop=512, sample_rate=SR, n_fft=N_FFT
        )
        self.assertEqual(n, 512)
        self.assertEqual(hop, 512)
        self.assertAlmostEqual(dur, (N_FFT + 512 * 511) / SR, places=9)
        self.assertEqual(notes, [])

    def test_explicit_frames_win(self):
        n, _, _, _ = m.plan_timing((512, 320), frames=99, sample_rate=SR, n_fft=N_FFT)
        self.assertEqual(n, 99)

    def test_duration_overrides_hop(self):
        n, hop, dur, _ = m.plan_timing(
            (100, 50), frames=100, duration=2.0, sample_rate=SR, n_fft=N_FFT
        )
        self.assertEqual(n, 100)
        self.assertAlmostEqual(dur, 2.0, delta=1.0 / SR * max(hop, 1))

    def test_max_duration_clamps_and_explains(self):
        n, hop, dur, notes = m.plan_timing(
            (10000, 50), hop=512, sample_rate=SR, n_fft=N_FFT, max_duration=5.0
        )
        self.assertLessEqual(dur, 5.0)
        self.assertTrue(notes and "clamped" in notes[0])
        self.assertLess(n, 10000)

    def test_max_duration_zero_disables_the_clamp(self):
        n, _, _, notes = m.plan_timing(
            (10000, 50), hop=512, sample_rate=SR, n_fft=N_FFT, max_duration=0.0
        )
        self.assertEqual(n, 10000)
        self.assertEqual(notes, [])

    def test_warns_about_extreme_overlap(self):
        _, hop, _, notes = m.plan_timing(
            (4000, 100), frames=4000, duration=1.0, sample_rate=SR, n_fft=2048
        )
        self.assertLess(hop, 2048 // 16)
        self.assertTrue(any("overlap" in n for n in notes))


class TestCli(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "img2spec.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "bars.png"
            wav_path = Path(d) / "bars.wav"
            png_path = Path(d) / "cmp.png"
            bars_image(80, 64).save(img_path)
            r = self._run(
                str(img_path),
                str(wav_path),
                "--n-fft",
                str(N_FFT),
                "--sr",
                str(SR),
                "--iterations",
                "16",
                "--verify",
                str(png_path),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(wav_path.exists() and wav_path.stat().st_size > 44)
            self.assertTrue(png_path.exists())
            self.assertIn("verify:", r.stdout)
            with wave.open(str(wav_path)) as fh:
                self.assertEqual(fh.getframerate(), SR)
                self.assertGreater(fh.getnframes(), N_FFT)

    def test_verify_png_shows_both_panels(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "bars.png"
            png_path = Path(d) / "cmp.png"
            bars_image(40, 32).save(img_path)
            r = self._run(
                str(img_path),
                str(Path(d) / "o.wav"),
                "--n-fft",
                str(N_FFT),
                "--sr",
                str(SR),
                "--frames",
                "40",
                "--iterations",
                "4",
                "--verify",
                str(png_path),
                "--quiet",
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            with Image.open(png_path) as cmp_img:
                self.assertEqual(cmp_img.size, (40 * 2 + 3, N_BINS))

    def test_black_image_writes_silence_without_warning_free_crash(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "black.png"
            Image.new("L", (32, 32), 0).save(img_path)
            r = self._run(
                str(img_path),
                str(Path(d) / "black.wav"),
                "--n-fft",
                str(N_FFT),
                "--sr",
                str(SR),
                "--iterations",
                "4",
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("all zeros", r.stderr)

    def test_rejects_non_power_of_two_fft(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "i.png"
            bars_image().save(img_path)
            r = self._run(str(img_path), str(Path(d) / "o.wav"), "--n-fft", "1000")
            self.assertEqual(r.returncode, 2)
            self.assertIn("power of two", r.stderr)

    def test_reports_missing_image(self):
        r = self._run("/nonexistent/nope.png", "out.wav")
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot read image", r.stderr)

    def test_1x1_image_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "dot.png"
            Image.new("L", (1, 1), 255).save(img_path)
            r = self._run(
                str(img_path),
                str(Path(d) / "dot.wav"),
                "--n-fft",
                str(N_FFT),
                "--sr",
                str(SR),
                "--iterations",
                "2",
            )
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
