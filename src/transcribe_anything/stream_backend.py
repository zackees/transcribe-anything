"""
Realtime streaming backend — faster-whisper + silero-vad (#124).

Slots into the ``streaming_fn=`` hook on :func:`server_app.create_app`
that landed in #123. The protocol skeleton there hands us a sync
iterator of PCM16-LE bytes and a :class:`StreamSession` we check for
cancellation between cycles; we yield ``{type, text, rev}`` event dicts
back over the WebSocket.

Module-level imports are kept lazy because ``faster-whisper`` /
``silero-vad`` only exist inside the streaming iso-env built by
:mod:`stream_reqs`. The host-side launcher imports this module just to
resolve the function pointer; the heavy imports run when the daemon
boots inside the iso-env.

Layout:

* :func:`faster_whisper_streaming_fn` — the entry point the WS handler
  calls. Drives the VAD + decoder loop and yields events.
* :class:`_Decoder` — small wrapper around ``WhisperModel`` so tests
  can monkey-patch the actual model away.
* :class:`_VadChunker` — silero-vad wrapper that takes PCM samples
  and returns ``(provisional_tail_samples, finalised_chunks)``.

All heavy work is hidden behind ``_lazy_imports()`` so the host process
can import this module without ``faster-whisper`` installed (the route
in ``server_app`` only instantiates the function pointer at startup).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator, Optional

from transcribe_anything.server_config import StreamCancelled, StreamSession

if TYPE_CHECKING:
    # numpy lives in the iso-env only.
    import numpy as _np_typeonly  # noqa: F401

LOG = logging.getLogger("transcribe_anything.stream_backend")

SAMPLE_RATE = 16000
DEFAULT_VAD_MIN_SILENCE_MS = 200
DEFAULT_MAX_WINDOW_SECONDS = 28.0
PROVISIONAL_OVERLAP_SECONDS = 2.0


def _lazy_imports():
    """Import the iso-env-only deps. Raises ImportError outside the iso-env."""
    import numpy as np  # type: ignore[import-not-found]
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    from silero_vad import (  # type: ignore[import-not-found]
        VADIterator,
        load_silero_vad,
    )

    return np, WhisperModel, VADIterator, load_silero_vad


class _Decoder:
    """Thin wrapper around ``faster_whisper.WhisperModel``.

    Exists so the streaming loop's test surface stays small: tests
    inject a fake ``_Decoder`` and assert the events that come out
    without having to mock the entire faster-whisper API.
    """

    def __init__(self, model_size: str = "small.en", device: str = "auto", compute_type: str = "default"):
        _, WhisperModel, _, _ = _lazy_imports()
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio, *, language: Optional[str] = None) -> str:
        """Return the concatenated transcript text for ``audio`` (1-D float32)."""
        segments, _info = self._model.transcribe(audio, language=language, beam_size=1, word_timestamps=False)
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text)
        return "".join(parts).strip()


class _VadChunker:
    """silero-vad wrapper that produces speech-segment boundaries."""

    def __init__(self, *, vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS):
        _, _, VADIterator, load_silero_vad = _lazy_imports()
        self._model = load_silero_vad()
        self._iter = VADIterator(self._model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=vad_min_silence_ms)
        self._buf_samples: list = []  # PCM samples (float32) since the last finalised cut
        self._in_speech = False

    def push(self, samples_f32):
        """Feed PCM samples (float32, 1-D). Returns a list of finalised chunks (numpy arrays)."""
        np, _, _, _ = _lazy_imports()
        finalised = []
        # silero-vad processes 512-sample windows at 16 kHz.
        window = 512
        offset = 0
        n = len(samples_f32)
        while offset + window <= n:
            chunk = samples_f32[offset : offset + window]
            event = self._iter(chunk, return_seconds=False)
            self._buf_samples.append(chunk)
            if isinstance(event, dict):
                if "start" in event:
                    self._in_speech = True
                if "end" in event and self._in_speech:
                    # Close the current chunk at this VAD boundary.
                    closed = np.concatenate(self._buf_samples) if self._buf_samples else np.zeros(0, dtype=np.float32)
                    finalised.append(closed)
                    self._buf_samples = []
                    self._in_speech = False
            offset += window
        # Any tail bytes that didn't form a full window are kept for next push.
        if offset < n:
            self._buf_samples.append(samples_f32[offset:])
        return finalised

    def provisional_tail(self):
        """Return the buffered (still-open) speech tail as a 1-D float32 array."""
        np, _, _, _ = _lazy_imports()
        if not self._buf_samples:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buf_samples)


def _pcm16_to_float32(buf: bytes):
    """Convert PCM16-LE bytes to mono float32 NumPy array."""
    np, _, _, _ = _lazy_imports()
    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def faster_whisper_streaming_fn(
    audio_iter: Iterator[bytes],
    *,
    session: StreamSession,
    config,
    hello: dict,
    decoder: Optional[_Decoder] = None,
    chunker: Optional[_VadChunker] = None,
) -> Iterator[dict]:
    """Streaming-fn entry point for the WS /v1/stream route.

    Loops:

    1. Pull whatever PCM has arrived since the last cycle out of
       ``audio_iter``.
    2. Convert to float32, hand it to silero-vad.
    3. For each finalised speech segment, run a full decode and yield
       a ``final`` event.
    4. Periodically (``stream_decode_interval_ms`` from config) yield a
       provisional ``partial`` event for the still-open tail.

    Raises :class:`StreamCancelled` when the session is cancelled.
    """
    np, _WhisperModel, _VADIterator, _load_silero_vad = _lazy_imports()  # noqa: F841

    model_size = (hello.get("model") if hello else None) or getattr(config, "model", None) or "small.en"
    language = hello.get("language") if hello else None

    if decoder is None:
        decoder = _Decoder(model_size=model_size)
    if chunker is None:
        chunker = _VadChunker()

    decode_interval_s = float(getattr(config, "stream_decode_interval_ms", 200)) / 1000.0
    rev = 0
    last_partial_text = ""
    last_partial_emit = 0.0

    yield {"type": "metrics", "backend": "faster-whisper", "model": model_size}

    while True:
        if session.is_cancelled:
            raise StreamCancelled("stream cancelled by client")

        # Drain whatever audio is currently buffered. The host-side
        # _audio_iterable returns when the queue is empty, so this loop
        # exits naturally and gives us a chance to emit a partial.
        any_audio = False
        for chunk_bytes in audio_iter:
            any_audio = True
            samples = _pcm16_to_float32(chunk_bytes)
            for closed in chunker.push(samples):
                if session.is_cancelled:
                    raise StreamCancelled("stream cancelled by client")
                if closed.size == 0:
                    continue
                text = decoder.transcribe(closed, language=language)
                if text:
                    rev += 1
                    last_partial_text = ""
                    yield {"type": "final", "text": text, "rev": rev}

        # Provisional partial every decode_interval_s.
        now = time.time()
        if now - last_partial_emit >= decode_interval_s:
            tail = chunker.provisional_tail()
            if tail.size >= int(0.5 * SAMPLE_RATE):  # ≥ 500ms of speech tail
                partial_text = decoder.transcribe(tail, language=language)
                if partial_text and partial_text != last_partial_text:
                    rev += 1
                    last_partial_text = partial_text
                    yield {"type": "partial", "text": partial_text, "rev": rev}
            last_partial_emit = now

        # No new audio AND no provisional fired — the route's
        # _audio_iterable returns empty when the WS pumper is idle. We
        # spin briefly so the cancellation flag is checked at the
        # configured cadence rather than busy-looping.
        if not any_audio:
            time.sleep(max(decode_interval_s / 4.0, 0.01))

        # If the client sent end_of_input the WS handler will close the
        # connection; that triggers cancel and we exit. No explicit EOF
        # check needed here — the cancellation flag is the contract.
