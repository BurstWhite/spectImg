#!/usr/bin/env python3
"""Generate the synthetic pattern used to demo and eyeball img2spec.

A real photo is a poor debugging target: when the reconstruction looks wrong
there is no way to tell whether the code or the picture is at fault.  This
pattern has known content - steady tones, a rising chirp, a burst and a ring -
so a mismatch is unambiguous.

    python tools/make_pattern.py demo/pattern.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 512
HEIGHT = 320

# (top row, bottom row, brightness) - horizontal bars are steady tones
BARS = ((20, 34, 255), (60, 70, 200), (110, 118, 170), (170, 176, 140), (240, 244, 110))


def build(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    a = np.zeros((height, width), dtype=np.uint8)

    for top, bottom, value in BARS:
        if top < height:
            a[top : min(bottom, height), :] = value

    # A chirp: one bright pixel column per time step, rising in frequency.
    for x in range(width):
        cy = int(height - 20 - (height - 60) * (x / width) ** 1.5)
        lo, hi = max(0, cy - 3), min(height, cy + 3)
        if lo < hi:
            a[lo:hi, x] = np.maximum(a[lo:hi, x], 220)

    img = Image.fromarray(a, mode="L")
    d = ImageDraw.Draw(img)
    d.rectangle(
        [int(0.18 * width), int(0.60 * height), int(0.26 * width), int(0.72 * height)],
        fill=230,
    )
    d.ellipse(
        [
            int(0.59 * width),
            int(0.44 * height),
            int(0.78 * width),
            int(0.69 * height),
        ],
        outline=210,
        width=6,
    )
    return img


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output", nargs="?", default="demo/pattern.png")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    args = p.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = build(args.width, args.height)
    img.save(path)
    print(f"wrote {path} ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
