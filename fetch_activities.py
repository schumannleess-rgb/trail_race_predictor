#!/usr/bin/env python3
"""CLI: Login → fetch running/trail_running activities → filter → select → download FIT files.

Usage:
    python fetch_activities.py                         # interactive mode
    python fetch_activities.py --type trail_running    # filter by type
    python fetch_activities.py --dist-min 10 --dist-max 50 --year 2025
    python fetch_activities.py --output records/fit    # custom output dir
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from login.garmin_login import garmin_login
from activities.selector import fetch_run_activities, filter_activities, interactive_select
from activities.downloader import download_fits


def main():
    parser = argparse.ArgumentParser(description="Fetch & download Garmin running activities")
    parser.add_argument("--email", help="Garmin email (or set GARMIN_EMAIL env)")
    parser.add_argument("--password", help="Garmin password (or set GARMIN_PASSWORD env)")
    parser.add_argument("--tokenstore", default=str(Path(__file__).parent / "tokens"))
    parser.add_argument("--type", dest="act_type", choices=["running", "trail_running"], help="Filter by type")
    parser.add_argument("--dist-min", type=float, help="Min distance (km)")
    parser.add_argument("--dist-max", type=float, help="Max distance (km)")
    parser.add_argument("--elev-min", type=float, help="Min elevation gain (m)")
    parser.add_argument("--elev-max", type=float, help="Max elevation gain (m)")
    parser.add_argument("--year", type=int, help="Filter by year")
    parser.add_argument("--month", type=int, help="Filter by month (1-12)")
    parser.add_argument("--output", "-o", default="records", help="Output directory for FIT files")
    parser.add_argument("--max-select", type=int, default=15, help="Max activities to select (default 15)")
    args = parser.parse_args()

    # 1. Login
    print("=" * 60)
    print("Step 1: Login")
    print("=" * 60)
    garmin = garmin_login(email=args.email, password=args.password, tokenstore=args.tokenstore)
    print(f"Logged in as: {garmin.display_name}\n")

    # 2. Fetch activities
    print("=" * 60)
    print("Step 2: Fetch running + trail_running activities")
    print("=" * 60)
    all_activities = fetch_run_activities(garmin)
    print(f"Fetched {len(all_activities)} running/trail_running activities\n")

    if not all_activities:
        print("No running activities found.")
        return

    # 3. Filter
    print("=" * 60)
    print("Step 3: Filter")
    print("=" * 60)
    filtered = filter_activities(
        all_activities,
        act_type=args.act_type,
        dist_min=args.dist_min, dist_max=args.dist_max,
        elev_min=args.elev_min, elev_max=args.elev_max,
        year=args.year, month=args.month,
    )
    print(f"After filtering: {len(filtered)} activities\n")

    if not filtered:
        print("No activities match the filter criteria.")
        return

    # 4. Interactive select
    print("=" * 60)
    print("Step 4: Select activities (max {})".format(args.max_select))
    print("=" * 60)
    selected = interactive_select(filtered, max_select=args.max_select)

    if not selected:
        print("No activities selected.")
        return

    # 5. Download FIT files
    print("\n" + "=" * 60)
    print("Step 5: Download FIT files")
    print("=" * 60)
    download_fits(garmin, selected, output_dir=args.output)


if __name__ == "__main__":
    main()
