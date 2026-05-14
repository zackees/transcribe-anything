"""Resolve any podcast-ish URL to a direct audio URL hostable for RunPod.

Designed to run on a residential-IP host (so YouTube / Spotify / paywalled
sources don't blanket-block the request) and upload the resulting audio to a
publicly-fetchable location that RunPod's datacenter workers can reach.

Usage:
    python3 runpod/resolve_and_host.py "<input-url>"
    # stdout: https://voice.vgh-usa.com/audio/transcribe-<uuid>.mp3

Pipeline:
    1. Classify input URL (direct media / listennotes / yt-dlp-supported / other)
    2. Direct media → pass through (no fetch needed; RunPod can grab it)
    3. listennotes / lnns.co → scrape the episode page for the publisher's CDN URL
    4. yt-dlp path → download, transcode to mp3, scp to AUDIO_HOST
    5. Print the final public URL on stdout

Required environment variables (set before invoking, e.g. via your shell
profile or a sourced env file):
    AUDIO_HOST_USER       SSH user on the audio-host (e.g. root)
    AUDIO_HOST            SSH host of the audio-host (hostname or IP)
    AUDIO_HOST_DIR        absolute path on the audio-host (e.g. /var/www/audio)
    AUDIO_PUBLIC_PREFIX   public URL prefix that maps to AUDIO_HOST_DIR
                          (e.g. https://audio.example.com/audio)
Optional:
    YTDLP_BIN             default: yt-dlp (prefers ~/.local/bin/yt-dlp if found)

The script will also auto-load $HOME/.vgh.env (a shell-style KEY=value file)
if it exists, so any of the above set there are picked up transparently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Best-effort loader for a shell-style KEY=value file. Doesn't override
    values already in os.environ (so explicit env vars win)."""
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


_load_env_file(Path.home() / ".vgh.env")


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(
            f"error: required environment variable {name} is not set.\n"
            f"Set it in your shell or in ~/.vgh.env. See script docstring for details.\n"
        )
        sys.exit(2)
    return val


AUDIO_HOST_USER = _require_env("AUDIO_HOST_USER")
AUDIO_HOST = _require_env("AUDIO_HOST")
AUDIO_HOST_DIR = _require_env("AUDIO_HOST_DIR")
AUDIO_PUBLIC_PREFIX = _require_env("AUDIO_PUBLIC_PREFIX")

# Prefer pipx-installed yt-dlp over the (often stale) apt one
_LOCAL_YTDLP = Path.home() / ".local" / "bin" / "yt-dlp"
YTDLP_BIN = os.environ.get(
    "YTDLP_BIN",
    str(_LOCAL_YTDLP) if _LOCAL_YTDLP.exists() else "yt-dlp",
)

DIRECT_AUDIO_CONTENT_TYPES = ("audio/", "application/octet-stream")
LISTENNOTES_HOSTS = ("lnns.co", "www.listennotes.com", "listennotes.com")
SPOTIFY_HOSTS = ("open.spotify.com", "spotify.link")


def log(msg: str) -> None:
    print(f"[resolve_and_host] {msg}", file=sys.stderr)


def classify(url: str) -> str:
    """Pick a resolution strategy based on the URL shape (cheap heuristic)."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if host in LISTENNOTES_HOSTS:
        return "listennotes"
    if host in SPOTIFY_HOSTS:
        return "spotify"
    if any(path.endswith(ext) for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus")):
        return "direct"
    return "ytdlp"


def head_content_type(url: str, timeout: int = 10) -> str | None:
    """Best-effort Content-Type detection via HEAD (some hosts reject HEAD; fall back)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.headers.get("Content-Type") or ""
    except Exception:
        return None


def is_direct_audio_url(url: str) -> bool:
    ct = head_content_type(url)
    if not ct:
        return False
    return any(ct.lower().startswith(p) for p in DIRECT_AUDIO_CONTENT_TYPES)


def resolve_listennotes(url: str) -> str:
    """Follow lnns.co / listennotes.com redirects to the episode page, then
    scrape for the publisher's audio URL.

    Listennotes embeds the canonical audio URL in a JSON-LD AudioObject
    `contentUrl` field in the episode page HTML.
    """
    log(f"resolving listennotes URL: {url}")
    # Follow redirects to the final HTML page
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 transcribe-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # 1) Try JSON-LD audio object (most reliable)
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        try:
            data = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if t == "PodcastEpisode" or t == "AudioObject":
                for key in ("contentUrl", "url", "audio"):
                    val = item.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        log(f"  found JSON-LD audio URL: {val}")
                        return val

    # 2) Fallback: scan for any *.mp3 in the page (loose)
    m = re.search(r'https?://[^\s"\'<>]+\.(?:mp3|m4a|wav|ogg)', html)
    if m:
        log(f"  found audio URL via mp3-scan: {m.group(0)}")
        return m.group(0)

    raise RuntimeError(
        "Could not extract a direct audio URL from the listennotes page. "
        "The episode may be hosted somewhere the scraper doesn't know about. "
        "Try the underlying podcast publisher's URL directly."
    )


def _fuzzy_match(target: str, candidates: list[str]) -> int | None:
    """Return the index of the candidate that best matches target.

    Uses normalized substring matching + word-overlap ratio. Returns None if
    nothing scores above the threshold.
    """
    def norm(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\w+", s or "") if len(w) > 2}

    target_words = norm(target)
    if not target_words:
        return None
    best_idx, best_score = None, 0.0
    for i, cand in enumerate(candidates):
        cand_words = norm(cand)
        if not cand_words:
            continue
        overlap = len(target_words & cand_words)
        score = overlap / max(len(target_words), 1)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx if best_score >= 0.5 else None


def resolve_spotify(url: str) -> str:
    """Resolve open.spotify.com episode URL to the publisher's CDN MP3.

    Spotify DRMs their audio, so we can't fetch it directly. But almost every
    podcast on Spotify also has a public RSS feed that publishes the same
    episode. Pipeline:

      1. Spotify oEmbed → episode title + show author
      2. iTunes Search API → RSS feed URL for the show
      3. Fetch RSS feed → find matching episode → return enclosure URL

    Spotify-exclusive shows (e.g. Joe Rogan during the exclusivity window)
    have no public feed and will fail here. That's a real Spotify limitation,
    not a bug in the resolver.
    """
    log(f"resolving Spotify URL: {url}")

    # Step 1: oEmbed
    oembed_url = "https://open.spotify.com/oembed?url=" + urllib.parse.quote(url)
    try:
        with urllib.request.urlopen(oembed_url, timeout=15) as resp:
            oembed = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        raise RuntimeError(f"Spotify oEmbed lookup failed: {e}")
    episode_title = oembed.get("title") or ""
    show_author = oembed.get("author_name") or oembed.get("provider_name") or ""
    log(f"  oEmbed: title={episode_title!r}, author={show_author!r}")
    if not episode_title:
        raise RuntimeError("Spotify oEmbed returned no title — can't look up the episode")

    # Step 2: iTunes Search to find the show's RSS feed
    # Many Spotify episode titles include the show name as a prefix or suffix.
    # We search using the show author (often the show name) first; fall back
    # to the episode title.
    search_term = show_author or episode_title
    itunes_url = (
        "https://itunes.apple.com/search?term="
        + urllib.parse.quote(search_term)
        + "&entity=podcast&limit=10&country=US"
    )
    log(f"  iTunes Search: {search_term!r}")
    try:
        with urllib.request.urlopen(itunes_url, timeout=15) as resp:
            itunes = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        raise RuntimeError(f"iTunes Search API failed: {e}")
    results = itunes.get("results") or []
    if not results:
        raise RuntimeError(
            f"iTunes returned no podcast matches for {search_term!r}. "
            "The show may be Spotify-exclusive."
        )
    # Pick the first result with a feedUrl (usually the right one)
    feed_url = None
    for r in results:
        if r.get("feedUrl"):
            feed_url = r["feedUrl"]
            log(f"  matched show: {r.get('collectionName')!r} → {feed_url}")
            break
    if not feed_url:
        raise RuntimeError("iTunes returned matches but none had a feedUrl")

    # Step 3: Fetch RSS feed and find the matching episode
    req = urllib.request.Request(
        feed_url,
        headers={"User-Agent": "Mozilla/5.0 transcribe-resolver/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rss = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch RSS feed {feed_url}: {e}")

    # Parse <item> entries. Don't bring in feedparser — the regex is enough
    # for the title + enclosure URL we need.
    items = re.findall(r"<item\b[^>]*>(.*?)</item>", rss, flags=re.DOTALL | re.IGNORECASE)
    log(f"  RSS feed has {len(items)} items")
    titles = []
    enclosures = []
    for item in items:
        title_m = re.search(
            r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
            item, flags=re.DOTALL | re.IGNORECASE,
        )
        enc_m = re.search(r'<enclosure[^>]*\burl=["\']([^"\']+)["\']', item, flags=re.IGNORECASE)
        titles.append(title_m.group(1).strip() if title_m else "")
        enclosures.append(enc_m.group(1).strip() if enc_m else "")

    idx = _fuzzy_match(episode_title, titles)
    if idx is None:
        log(f"  no fuzzy match for episode title — trying first item with audio")
        for i, enc in enumerate(enclosures):
            if enc:
                idx = i
                log(f"  using first audio item: {titles[i]!r}")
                break
    if idx is None or not enclosures[idx]:
        raise RuntimeError(
            f"Couldn't find episode in RSS feed. Searched for: {episode_title!r}\n"
            f"Available titles (first 5): {titles[:5]}"
        )

    log(f"  matched episode: {titles[idx]!r}")
    log(f"  enclosure URL: {enclosures[idx]}")
    return enclosures[idx]


def download_with_ytdlp(url: str, out_dir: Path) -> Path:
    """Run yt-dlp to fetch the best audio stream and convert to mp3."""
    log(f"yt-dlp downloading: {url}")
    out_template = str(out_dir / "audio.%(ext)s")
    cmd = [
        YTDLP_BIN,
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",  # best quality
        "--no-playlist",
        "-o", out_template,
        url,
    ]
    log(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # yt-dlp drops audio.mp3 (or similar) in the dir
    for cand in out_dir.glob("audio.*"):
        if cand.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac"):
            return cand
    raise RuntimeError(f"yt-dlp succeeded but no audio file found in {out_dir}")


def download_direct(url: str, out_dir: Path) -> Path:
    """Download a direct media URL via curl (handles redirects, large files)."""
    log(f"downloading direct media: {url}")
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".mp3"
    dest = out_dir / f"audio{suffix}"
    subprocess.run(
        ["curl", "-sSL", "--retry", "3", "-o", str(dest), url],
        check=True,
    )
    return dest


def upload(local_path: Path) -> str:
    """scp the local audio file to AUDIO_HOST and return its public URL."""
    remote_name = f"transcribe-{uuid.uuid4().hex[:12]}{local_path.suffix.lower()}"
    remote_target = f"{AUDIO_HOST_USER}@{AUDIO_HOST}:{AUDIO_HOST_DIR}/{remote_name}"
    log(f"scp {local_path} -> {remote_target}")
    subprocess.run(
        ["scp", "-q", "-o", "ConnectTimeout=10", str(local_path), remote_target],
        check=True,
    )
    public_url = f"{AUDIO_PUBLIC_PREFIX}/{remote_name}"
    return public_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Input URL (Spotify, YouTube, podcast page, lnns.co, direct mp3, etc.)")
    parser.add_argument(
        "--passthrough-direct",
        action="store_true",
        help="If the URL is already direct audio, print it unchanged instead of mirroring through the audio host.",
    )
    args = parser.parse_args()

    url = args.url.strip()
    strategy = classify(url)

    # If classify says direct, double-check via HEAD before short-circuiting.
    if strategy == "direct" or is_direct_audio_url(url):
        if args.passthrough_direct:
            log(f"direct audio URL; printing unchanged")
            print(url)
            return 0
        log("direct audio URL; mirroring through audio host for stable serving")
        strategy = "direct"

    with tempfile.TemporaryDirectory(prefix="transcribe-resolve-") as tmpdir:
        out_dir = Path(tmpdir)
        if strategy == "listennotes":
            resolved = resolve_listennotes(url)
            local_path = download_direct(resolved, out_dir)
        elif strategy == "spotify":
            resolved = resolve_spotify(url)
            local_path = download_direct(resolved, out_dir)
        elif strategy == "direct":
            local_path = download_direct(url, out_dir)
        else:
            local_path = download_with_ytdlp(url, out_dir)

        public_url = upload(local_path)
        print(public_url)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        log(f"subprocess failed: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
