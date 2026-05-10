"""Fetch, filter, and interactively select running/trail_running activities."""

import time
from datetime import datetime

RUN_TYPES = {"running", "trail_running", "trail_running_v2"}


def fetch_run_activities(garmin, batch_size=100, max_batches=20):
    """Fetch all running + trail_running activities from Garmin Connect.

    Args:
        garmin: Authenticated Garmin client.
        batch_size: Activities per API call.
        max_batches: Safety limit on pagination.

    Returns:
        List of dicts with keys: id, name, distance_km, elevation_m, date, year, month, type.
    """
    results = []
    for batch in range(max_batches):
        offset = batch * batch_size
        raw = garmin.get_activities(offset, batch_size)
        if not raw:
            break

        for a in raw:
            atype = a.get("activityType", {}).get("typeKey", "")
            if atype not in RUN_TYPES:
                continue
            dt = a.get("startTimeLocal", "")
            results.append({
                "id": a["activityId"],
                "name": a.get("activityName", ""),
                "distance_km": round((a.get("distance") or 0) / 1000, 2),
                "elevation_m": a.get("elevationGain") or 0,
                "date": dt[:10],
                "year": int(dt[:4]) if len(dt) >= 4 else 0,
                "month": int(dt[5:7]) if len(dt) >= 7 else 0,
                "type": atype,
            })

        if len(raw) < batch_size:
            break
        time.sleep(0.3)

    results.sort(key=lambda x: x["date"], reverse=True)
    return results


def filter_activities(activities, *, act_type=None, dist_min=None, dist_max=None,
                      elev_min=None, elev_max=None, year=None, month=None):
    """Filter activities by criteria. All params optional."""

    def match(a):
        if act_type and a["type"] != act_type:
            return False
        if dist_min is not None and a["distance_km"] < dist_min:
            return False
        if dist_max is not None and a["distance_km"] > dist_max:
            return False
        if elev_min is not None and a["elevation_m"] < elev_min:
            return False
        if elev_max is not None and a["elevation_m"] > elev_max:
            return False
        if year is not None and a["year"] != year:
            return False
        if month is not None and a["month"] != month:
            return False
        return True

    return [a for a in activities if match(a)]


def interactive_select(activities, max_select=15):
    """Display activities table, let user pick up to max_select by index.

    Args:
        activities: List of activity dicts (already filtered).
        max_select: Maximum number of activities to select.

    Returns:
        List of selected activity dicts.
    """
    if not activities:
        print("No activities to select.")
        return []

    # Display table
    print(f"\n{'#':>3}  {'Date':<10} {'Type':<16} {'Dist(km)':>9} {'Elev(m)':>8}  Name")
    print("-" * 80)
    for i, a in enumerate(activities):
        print(f"{i:>3}  {a['date']:<10} {a['type']:<16} {a['distance_km']:>9.1f} {a['elevation_m']:>8.0f}  {a['name'][:40]}")
    print("-" * 80)
    print(f"Total: {len(activities)} activities")

    while True:
        raw = input(f"\nSelect activities (comma-separated indices, max {max_select}, 'q' to quit): ").strip()
        if raw.lower() == "q":
            return []

        try:
            indices = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("Invalid input. Enter numbers separated by commas.")
            continue

        bad = [i for i in indices if i < 0 or i >= len(activities)]
        if bad:
            print(f"Index out of range: {bad}. Valid: 0-{len(activities)-1}")
            continue

        if len(indices) > max_select:
            print(f"Too many selected ({len(indices)}). Max is {max_select}.")
            continue

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique.append(i)

        selected = [activities[i] for i in unique]
        print(f"\nSelected {len(selected)} activities:")
        for a in selected:
            print(f"  [{a['id']}] {a['date']} {a['distance_km']:.1f}km {a['elevation_m']:.0f}m {a['name'][:40]}")
        return selected
