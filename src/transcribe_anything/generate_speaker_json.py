"""
Test the parse speaker module.
"""

from dataclasses import dataclass
from typing import Optional
from warnings import warn


@dataclass
class Chunk:
    """Chunk of text."""

    speaker: str
    timestamp_start: float
    timestamp_end: float
    text: str
    reason: Optional[str] = None

    def to_json(self) -> dict:
        """Convert to json."""
        out = {
            "speaker": self.speaker,
            "timestamp": [self.timestamp_start, self.timestamp_end],
            "text": self.text,
            "reason": self.reason,
        }
        return out


def can_combine(chunk1: Chunk, chunk2: Chunk) -> bool:
    """Check if two chunks can be combined."""
    return chunk1.speaker == chunk2.speaker and abs(chunk1.timestamp_end - chunk2.timestamp_start) <= 0.1


def reduce(dat: list[Chunk]) -> list[Chunk]:
    """Reduce a list of chunks."""
    out: list[Chunk] = []
    for chunk in dat:
        if not out:
            chunk.reason = "beginning"
            out.append(chunk)
            continue
        last_chunk = out[-1]
        if not can_combine(last_chunk, chunk):
            chunk.reason = "speaker-switch" if last_chunk.speaker != chunk.speaker else "timeout"
            out.append(chunk)
            continue
        # combine
        out[-1] = Chunk(
            last_chunk.speaker,
            last_chunk.timestamp_start,
            chunk.timestamp_end,
            last_chunk.text + " " + chunk.text,
            last_chunk.reason,
        )

    return out


def generate_speaker_json(json_data: dict) -> list[dict]:
    """Generate a speaker json."""
    speaker_chunks = json_data.get("speakers", {})
    if not speaker_chunks:
        warn("No speaker data found.")
        return []
    # convert to a list of chunks
    chunks: list[Chunk] = []
    for chunk in speaker_chunks:
        try:
            speaker = chunk["speaker"]
            timestamp_start = float(chunk["timestamp"][0])
            timestamp_end = float(chunk["timestamp"][1])
            text = chunk["text"]
            chunks.append(Chunk(speaker, timestamp_start, timestamp_end, text))
        except KeyError:
            import warnings

            warnings.warn(f"Invalid chunk: {chunk}")
    reduced = reduce(chunks)
    out = [chunk.to_json() for chunk in reduced]
    return out
