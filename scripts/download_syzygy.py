#!/usr/bin/env python3
"""Download 3-4-5 piece Syzygy endgame tablebases from the Lichess mirror.

Usage
-----
  # Download to default location (data/syzygy/):
  python scripts/download_syzygy.py

  # Download to a custom path:
  python scripts/download_syzygy.py --path /mnt/tablebases/syzygy

  # Override with environment variable (useful in CI/Docker):
  SYZYGY_PATH=/opt/syzygy python scripts/download_syzygy.py

  # Verify existing tables without downloading:
  python scripts/download_syzygy.py --verify-only

  # Force re-download (ignore existing files):
  python scripts/download_syzygy.py --force

Design
------
- Scrapes the Lichess open-source mirror for the canonical file list.
- Downloads only missing or incomplete files (resume-safe).
- Validates each file against the Content-Length advertised by the server.
- Parallelises downloads with a configurable worker count.
- Zero mandatory third-party dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# --- Constants ---
MIRROR_BASE = "https://tablebase.lichess.ovh/tables/standard"
WDL_DIR = f"{MIRROR_BASE}/3-4-5-wdl/"
DTZ_DIR = f"{MIRROR_BASE}/3-4-5-dtz/"

USER_AGENT = "Moray-SyzygyDownloader/1.0"

DEFAULT_SYZYGY_DIR = Path("data/syzygy")

# Retry / concurrency knobs
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubled each retry
DEFAULT_WORKERS = 4


# --- Data ---
@dataclass(frozen=True)
class RemoteFile:
    """A single file on the mirror, with its expected byte-size."""

    url: str
    name: str
    size: int  # expected Content-Length in bytes


# --- Mirror scraping ---
def _fetch_url(url: str) -> str:
    """Fetch a URL and return its decoded body."""
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _parse_directory_listing(index_url: str) -> list[RemoteFile]:
    """Parse an nginx autoindex HTML listing into RemoteFile entries."""
    body = _fetch_url(index_url)

    # Format: <a href="KBBBvK.rtbw">KBBBvK.rtbw</a>  07-Dec-2013 02:14  739600
    pattern = re.compile(
        r'<a\s+href="(?P<name>[^"]+\.rtb[wz])">'
        r"[^<]+</a>\s+"
        r"\d{2}-\w{3}-\d{4}\s+\d{2}:\d{2}\s+"
        r"(?P<size>\d+)"
    )

    files: list[RemoteFile] = []
    for match in pattern.finditer(body):
        name = html.unescape(match.group("name"))
        size = int(match.group("size"))
        files.append(RemoteFile(url=index_url + name, name=name, size=size))

    return files


def discover_remote_files() -> list[RemoteFile]:
    """Return the full manifest of WDL + DTZ files on the mirror."""
    print("Discovering files on the Lichess Syzygy mirror ...")
    wdl = _parse_directory_listing(WDL_DIR)
    dtz = _parse_directory_listing(DTZ_DIR)
    manifest = wdl + dtz
    total_bytes = sum(f.size for f in manifest)
    print(
        f"  Found {len(wdl)} WDL + {len(dtz)} DTZ = {len(manifest)} files "
        f"({total_bytes / 1e9:.2f} GB)"
    )
    return manifest


# --- Download logic ---
def _needs_download(local_path: Path, expected_size: int, *, force: bool) -> bool:
    """Return True if the file should be (re-)downloaded."""
    if force:
        return True
    if not local_path.exists():
        return True
    return local_path.stat().st_size != expected_size


def _download_one(remote: RemoteFile, dest_dir: Path, *, force: bool) -> str | None:
    """Download a single file with retries.  Returns an error message or None."""
    local_path = dest_dir / remote.name

    if not _needs_download(local_path, remote.size, force=force):
        return None  # already present and correct size

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(  # noqa: S310
                remote.url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(  # noqa: S310
                req, timeout=120
            ) as resp:
                data = resp.read()

            local_path.write_bytes(data)

            # Validate size
            if local_path.stat().st_size != remote.size:
                msg = (
                    f"{remote.name}: size mismatch "
                    f"(got {local_path.stat().st_size}, expected {remote.size})"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                return msg

            return None  # success

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return f"{remote.name}: {exc}"

    return f"{remote.name}: exhausted retries"


def download_tables(
    dest_dir: Path,
    manifest: list[RemoteFile],
    *,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> bool:
    """Download all files in *manifest* into *dest_dir*.

    Returns True if all files are present and valid, False on any error.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    to_download = [
        f for f in manifest if _needs_download(dest_dir / f.name, f.size, force=force)
    ]

    if not to_download:
        print("All Syzygy tablebase files are already present  -  nothing to do.")
        return True

    total_bytes = sum(f.size for f in to_download)
    print(
        f"Downloading {len(to_download)} files "
        f"({total_bytes / 1e6:.1f} MB) with {workers} workers ..."
    )

    errors: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, f, dest_dir, force=force): f for f in to_download
        }
        for future in as_completed(futures):
            done += 1
            err = future.result()
            if err:
                errors.append(err)
                print(f"  [{done}/{len(to_download)}] FAIL  {err}")
            else:
                rf = futures[future]
                print(
                    f"  [{done}/{len(to_download)}] OK    {rf.name} "
                    f"({rf.size / 1e6:.1f} MB)"
                )

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("All files downloaded successfully.")
    return True


# --- Verification ---
def verify_tables(dest_dir: Path, manifest: list[RemoteFile]) -> bool:
    """Check that every file in *manifest* exists locally with the right size."""
    missing: list[str] = []
    bad_size: list[str] = []

    for rf in manifest:
        local = dest_dir / rf.name
        if not local.exists():
            missing.append(rf.name)
        elif local.stat().st_size != rf.size:
            bad_size.append(
                f"{rf.name} (local={local.stat().st_size}, expected={rf.size})"
            )

    ok = not missing and not bad_size
    if ok:
        print(f"Verification passed: all {len(manifest)} files present and correct.")
    else:
        if missing:
            print(f"{len(missing)} missing file(s):")
            for m in missing[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        if bad_size:
            print(f"{len(bad_size)} file(s) with wrong size:")
            for b in bad_size[:10]:
                print(f"  - {b}")
    return ok


# --- Public helper for Nox / programmatic use ---
def ensure_syzygy(
    path: Path | str | None = None,
    *,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Ensure Syzygy tables are present at *path*.

    Resolves the path in priority order:
      1. Explicit *path* argument
      2. ``SYZYGY_PATH`` environment variable
      3. ``data/syzygy/`` (project default)

    Returns the resolved directory path.
    Raises ``SystemExit`` on download failure.
    """
    if path is not None:
        dest = Path(path)
    else:
        dest = Path(os.environ.get("SYZYGY_PATH", str(DEFAULT_SYZYGY_DIR)))

    manifest = discover_remote_files()

    if verify_tables(dest, manifest):
        return dest

    ok = download_tables(dest, manifest, force=force, workers=workers)
    if not ok:
        print(
            "ERROR: Syzygy download incomplete  -  see failures above.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return dest


# --- CLI ---
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download 3-4-5 piece Syzygy endgame tablebases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(f"Destination directory (default: $SYZYGY_PATH or {DEFAULT_SYZYGY_DIR})"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all files even if they already exist.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files  -  do not download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel download threads (default: {DEFAULT_WORKERS}).",
    )
    return parser


def main() -> None:
    """Download and verify Syzygy endgame tablebases.

    Parses command line arguments, discovers remote files, verifies
    existing tables, and downloads missing ones.
    """
    args = _build_parser().parse_args()

    dest = Path(args.path or os.environ.get("SYZYGY_PATH") or str(DEFAULT_SYZYGY_DIR))

    manifest = discover_remote_files()

    if args.verify_only:
        ok = verify_tables(dest, manifest)
        raise SystemExit(0 if ok else 1)

    if verify_tables(dest, manifest):
        return

    ok = download_tables(dest, manifest, force=args.force, workers=args.workers)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
