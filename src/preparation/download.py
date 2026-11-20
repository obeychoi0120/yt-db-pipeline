"""Download manifest videos with yt-dlp into preparation.data_root."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from artifact_io import write_json

DOWNLOAD_DEFAULTS = {
    "height": 360,
    "cookies_file": None,
    "sleep_seconds": 10,
    "rate_limit_sleep_seconds": 600,
    "max_rate_limit_retries": 3,
    # Corporate networks: an HTTP(S) proxy, and either the proxy's CA certificate or no
    # certificate checks at all. None/False leaves yt-dlp's own behaviour.
    "proxy": None,
    "ca_cert_file": None,
    "no_check_certificates": False,
}
METADATA_FIELDS = (
    "id", "title", "channel", "channel_id", "uploader", "upload_date", "duration",
    "view_count", "categories", "tags", "language", "width", "height", "fps", "webpage_url",
    "description",
)


def download_settings(config) -> dict:
    values = {**DOWNLOAD_DEFAULTS, **((config.get("preparation") or {}).get("download") or {})}
    if type(values["height"]) is not int or values["height"] <= 0:
        raise ValueError("preparation.download.height must be a positive integer")
    if values["ca_cert_file"] and not Path(values["ca_cert_file"]).expanduser().is_file():
        raise ValueError(f"preparation.download.ca_cert_file not found: {values['ca_cert_file']}")
    if values["cookies_file"] and not Path(values["cookies_file"]).expanduser().is_file():
        raise ValueError(f"preparation.download.cookies_file not found: {values['cookies_file']}")
    return values


def ytdlp_command(url: str, video_path: Path, metadata_stem: Path, settings: dict) -> list[str]:
    command = [
        "yt-dlp", "--ignore-config", "--no-playlist", "--continue", "--no-overwrites",
        "--retries", "3", "--fragment-retries", "3", "--socket-timeout", "30",
        "--write-info-json", "-f", format_selector(settings["height"]),
        "--remux-video", "mp4",
        "-o", str(video_path.with_suffix(".%(ext)s")),
        "-o", f"infojson:{metadata_stem}",
    ]
    if settings["cookies_file"]:
        command += ["--cookies", str(Path(settings["cookies_file"]).expanduser())]
    if settings["proxy"]:
        command += ["--proxy", settings["proxy"]]
    if settings["no_check_certificates"]:
        command.append("--no-check-certificates")
    elif settings["ca_cert_file"]:
        # yt-dlp trusts only certifi's bundle unless told to use the system store, which
        # SSL_CERT_FILE (set in ytdlp_env) then points at the proxy's certificate.
        command += ["--compat-options", "no-certifi"]
    return [*command, url]


def ytdlp_env(settings: dict) -> dict[str, str]:
    """yt-dlp's environment; the proxy also reaches its helpers (deno for JS challenges)."""
    env = dict(os.environ)
    if settings["proxy"]:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = settings["proxy"]
    if settings["ca_cert_file"] and not settings["no_check_certificates"]:
        path = str(Path(settings["ca_cert_file"]).expanduser())
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "DENO_CERT"):
            env[key] = path
    return env


def shown_proxy(proxy: str | None) -> str | None:
    """The proxy for logs, without a user name or password."""
    if not proxy or "@" not in proxy:
        return proxy
    scheme, _, rest = proxy.rpartition("://")
    return f"{scheme}://***@{rest.split('@', 1)[1]}" if scheme else "***@" + proxy.split("@", 1)[1]


def format_selector(height: int) -> str:
    """Video only: the requested height first, then the smallest larger one, then anything.

    Keyframes are always scaled and padded to the configured resolution afterwards, so a
    larger source gives the same keyframes.
    """
    return (
        f"bv[ext=mp4][height={height}]/b[ext=mp4][height={height}]/"
        f"bv[height={height}]/"
        f"bv[ext=mp4][height>{height}]/bv[height>{height}]/"
        "bv[ext=mp4]/bv/b"
    )


def valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
             "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
        data = json.loads(result.stdout)
        return (result.returncode == 0
                and any(s.get("codec_type") == "video" for s in data.get("streams", []))
                and float(data.get("format", {}).get("duration", 0)) > 0)
    except (ValueError, subprocess.TimeoutExpired):
        return False


def run_ytdlp(command, env=None) -> tuple[int, bool]:
    """Relay output live while keeping only a small tail for rate-limit detection."""
    rate_limited = False
    tail = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    ) as process:
        try:
            while chunk := process.stdout.read1(8192):
                output = decoder.decode(chunk)
                sys.stdout.write(output)
                sys.stdout.flush()
                message = (tail + output).lower()
                rate_limited |= (
                    "http error 429" in message
                    or "too many requests" in message
                    or ("sign in to confirm" in message and "not a bot" in message)
                )
                tail = message[-1024:]
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()
            return process.wait(), rate_limited
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def _cooldown(seconds: int) -> None:
    for remaining in range(seconds, 0, -30):
        minutes, rest = divmod(remaining, 60)
        print(f"[COOLDOWN] {minutes:02d}:{rest:02d} until retry", flush=True)
        time.sleep(min(30, remaining))


def _store_metadata(info_path: Path, metadata_path: Path, row: dict) -> None:
    """Keep a compact metadata file; the full yt-dlp info JSON is removed."""
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    write_json(
        metadata_path,
        {"content_id": row["content_id"], "url": row["url"],
         **{key: info.get(key) for key in METADATA_FIELDS}},
    )
    info_path.unlink()


def download_videos(context, rows, *, settings) -> dict:
    """Download each manifest row that has no valid video yet; never overwrite a file."""
    videos_dir, metadata_dir = context.videos_dir, context.metadata_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    counts = {"present": 0, "downloaded": 0, "failed": 0}
    failures = []
    for index, row in enumerate(rows, 1):
        cid = row["content_id"]
        path = videos_dir / f"{cid}.mp4"
        label = f"[{index}/{len(rows)}] {cid}"
        if valid_video(path):
            counts["present"] += 1
            continue
        if path.exists():
            print(f"{label}: invalid existing file; inspect or remove it: {path}", flush=True)
            counts["failed"] += 1
            failures.append({"content_id": cid, "reason": "invalid_existing_file"})
            continue
        command = ytdlp_command(row["url"], path, metadata_dir / cid, settings)
        env = ytdlp_env(settings)
        print(f"\n{label}: downloading", flush=True)
        delay = settings["rate_limit_sleep_seconds"]
        ok = False
        for attempt in range(settings["max_rate_limit_retries"] + 1):
            returncode, rate_limited = run_ytdlp(command, env)
            ok = returncode == 0 and valid_video(path)
            if ok or not rate_limited or attempt == settings["max_rate_limit_retries"]:
                break
            print(f"{label}: rate limit or bot check; retrying in {delay // 60} min", flush=True)
            _cooldown(delay)
            delay = min(delay * 2, 3600)
        if ok:
            _store_metadata(metadata_dir / f"{cid}.info.json", metadata_dir / f"{cid}.json", row)
            counts["downloaded"] += 1
            print(f"{label}: OK", flush=True)
            time.sleep(settings["sleep_seconds"])
        else:
            counts["failed"] += 1
            failures.append({
                "content_id": cid,
                "reason": "rate_limited" if rate_limited else "download_failed",
            })
            print(f"{label}: FAILED; see output above", flush=True)
    return {**counts, "failures": failures}
