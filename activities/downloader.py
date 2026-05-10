"""Download FIT files for selected activities."""

import re
import time
from pathlib import Path


def _safe_filename(name, activity_id):
    """Sanitize activity name for use as filename."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > 80:
        name = name[:80]
    return f"{name or 'Unnamed'}_{activity_id}.fit"


def download_fits(garmin, activities, output_dir="records"):
    """Download FIT files for a list of activities.

    Args:
        garmin: Authenticated Garmin client.
        activities: List of activity dicts (must have 'id' and 'name').
        output_dir: Directory to save FIT files.

    Returns:
        Tuple of (downloaded_count, skipped_count, failed_count).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0

    for i, a in enumerate(activities, 1):
        aid = a["id"]
        fname = _safe_filename(a["name"], aid)
        fpath = out / fname

        if fpath.exists():
            print(f"[{i}/{len(activities)}] SKIP (exists): {fname}")
            skipped += 1
            continue

        print(f"[{i}/{len(activities)}] Downloading {aid}...", end=" ")
        try:
            data = garmin.download_activity(aid, dl_fmt=garmin.ActivityDownloadFormat.ORIGINAL)
            with open(fpath, "wb") as f:
                f.write(data)
            print(f"OK ({len(data)} bytes)")
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        time.sleep(0.5)

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Output: {out.resolve()}")
    return downloaded, skipped, failed
