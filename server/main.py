"""HTTP service wrapping img2spec.py: image in, WAV out.

Endpoints
---------
GET  /api/health   liveness probe
POST /api/convert  multipart: image + turnstile token + options -> audio/wav

The WAV is synthesized in memory and returned with ``Cache-Control: no-store``;
nothing is written to disk and no request/response bodies are kept server-side.

Environment
-----------
TURNSTILE_SECRET_KEY   Cloudflare Turnstile secret key. Defaults to the
                       official always-pass test key so local dev works;
                       set the real key in production.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import img2spec  # noqa: E402

# Cloudflare's public test keys: always-pass widget + always-pass verification.
TEST_SECRET_KEY = "1x0000000000000000000000000000000AA"
SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 30_000_000  # PIL's own guard value; rejects decompression bombs

ALLOWED_SR = (8000, 11025, 16000, 22050, 32000, 44100, 48000)

app = FastAPI(title="img2spec service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # no auth anywhere; browsers still enforce same-origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Synthesis-Seconds", "X-Synthesis-Notes"],
)


async def verify_turnstile(token: str, remote_ip: str | None) -> None:
    secret = os.environ.get("TURNSTILE_SECRET_KEY", TEST_SECRET_KEY)
    if not token:
        raise HTTPException(400, "missing turnstile token")
    data = {"secret": secret, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(SITEVERIFY_URL, data=data)
            r.raise_for_status()
            outcome = r.json()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"turnstile verification unreachable: {exc}") from exc
    if not outcome.get("success"):
        raise HTTPException(403, "turnstile verification failed")


def _clamp(name: str, value: float, lo: float, hi: float) -> float:
    if not lo <= value <= hi:
        raise HTTPException(400, f"{name} must be between {lo} and {hi}")
    return value


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/api/convert")
async def convert(
    image: UploadFile = File(...),
    turnstile_token: str = Form(...),
    freq_scale: str = Form("linear"),
    duration: float | None = Form(None),
    min_db: float = Form(-80.0),
    gamma: float = Form(1.0),
    iterations: int = Form(64),
    sr: int = Form(22050),
    n_fft: int = Form(2048),
) -> Response:
    await verify_turnstile(
        turnstile_token, None
    )  # remote IP: behind Cloudflare the socket IP is wrong anyway

    payload = await image.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 12 MB)")
    if not payload:
        raise HTTPException(400, "empty upload")

    if freq_scale not in img2spec.FREQ_SCALES:
        raise HTTPException(400, f"freq_scale must be one of {img2spec.FREQ_SCALES}")
    if sr not in ALLOWED_SR:
        raise HTTPException(400, f"sr must be one of {ALLOWED_SR}")
    if n_fft < 8 or n_fft & (n_fft - 1):
        raise HTTPException(400, "n_fft must be a power of two >= 8")
    _clamp("min_db", min_db, -120.0, 0.0)
    _clamp("gamma", gamma, 0.1, 5.0)
    _clamp("iterations", iterations, 0, 256)
    if duration is not None:
        _clamp("duration", duration, 1.0, 60.0)

    try:
        img = Image.open(io.BytesIO(payload))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"cannot decode image: {exc}") from exc
    if img.width < 1 or img.height < 1 or img.width * img.height > MAX_PIXELS:
        raise HTTPException(400, "image dimensions out of range")

    n_frames, hop, _, notes = img2spec.plan_timing(
        img.size,
        duration=duration,
        sample_rate=sr,
        n_fft=n_fft,
        max_duration=60.0,
    )
    with img:
        mag = img2spec.image_to_magnitude(
            img,
            n_frames=n_frames,
            n_fft=n_fft,
            sample_rate=sr,
            freq_scale=freq_scale,
            min_db=min_db,
            gamma=gamma,
        )

    y, _ = img2spec.griffin_lim(
        mag, n_fft=n_fft, hop=hop, n_iter=iterations, progress=False
    )
    y = img2spec.normalize(y, "peak", -1.0)

    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype="PCM_16", format="WAV")
    wav_bytes = buf.getvalue()

    headers = {
        "Cache-Control": "no-store",
        "X-Synthesis-Seconds": f"{len(y) / sr:.2f}",
        "X-Synthesis-Notes": "; ".join(notes),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
