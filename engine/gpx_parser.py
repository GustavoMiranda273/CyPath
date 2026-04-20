"""
CyPath GPX Parser
Parses a GPX file exported from Strava (or any compatible device) and
estimates the Training Stress Score (TSS) for the session.

Supported data sources:
  - GPS coordinates + timestamps  → distance (km) and duration (seconds)
  - Elevation data                → total climbing (metres)
  - Heart rate extensions         → avg HR → TRIMP-based TSS estimate
  - No HR data                    → speed-based intensity estimate

TSS formulae
────────────
With heart rate:
    intensity = avg_hr / ESTIMATED_MAX_HR   (185 bpm — reasonable cycling average)
    TSS = (duration_hrs × intensity²) × 100

Without heart rate (speed proxy):
    intensity = lookup from average speed bands
    TSS = (duration_hrs × intensity²) × 100

These are estimates, not precise power-based calculations.
The confirmation screen lets the user adjust the value before logging.

Author: Gustavo Miranda
References: Borresen & Lambert (2009); Coggan & Allen (2010)
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


# ─── Constants ───────────────────────────────────────────────────────────────

# Assumed maximum heart rate for intensity estimation (bpm).
# 185 is a conservative average for recreational cyclists.
ESTIMATED_MAX_HR: float = 185.0

# Speed → intensity factor mapping (km/h → IF).
# Used as a fallback when no heart rate data is available.
SPEED_INTENSITY_BANDS = [
    (15, 0.45),   # < 15 km/h — very easy spin
    (20, 0.55),   # 15–20 km/h — easy
    (25, 0.65),   # 20–25 km/h — moderate
    (30, 0.75),   # 25–30 km/h — brisk
    (35, 0.85),   # 30–35 km/h — hard
    (999, 0.95),  # > 35 km/h — very hard
]

# GPX XML namespaces used by Garmin / Strava for heart rate extensions.
GPX_NAMESPACES = {
    "gpx":   "http://www.topografix.com/GPX/1/1",
    "gpxtpx":"http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
    "ns3":   "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
}


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class GPXSummary:
    """Parsed metrics from a single GPX activity."""
    distance_km:       float          # Total distance
    duration_seconds:  float          # Moving time
    elevation_gain_m:  float          # Total climbing
    avg_hr_bpm:        Optional[float] # Average heart rate (None if unavailable)
    estimated_tss:     float          # Calculated TSS estimate
    method:            str            # "heart_rate" or "speed"
    activity_name:     str            # From GPX <name> tag if present

    @property
    def duration_str(self) -> str:
        """Human-readable duration string, e.g. '1h 24m'."""
        h = int(self.duration_seconds // 3600)
        m = int((self.duration_seconds % 3600) // 60)
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"

    @property
    def avg_speed_kmh(self) -> float:
        """Average speed in km/h."""
        hrs = self.duration_seconds / 3600
        return round(self.distance_km / hrs, 1) if hrs > 0 else 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points (km)."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(a, 1.0)))


def _speed_intensity(avg_speed_kmh: float) -> float:
    """Return an intensity factor (0–1) based on average speed."""
    for threshold, factor in SPEED_INTENSITY_BANDS:
        if avg_speed_kmh < threshold:
            return factor
    return 0.95


def _find_hr(trkpt: ET.Element) -> Optional[float]:
    """
    Extract heart rate from a <trkpt> element.
    Handles Garmin/Strava namespace variants.
    """
    # Try various namespace paths that different devices use.
    hr_paths = [
        ".//{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr",
        ".//{http://www.garmin.com/xmlschemas/TrackPointExtension/v2}hr",
        ".//hr",
    ]
    for path in hr_paths:
        el = trkpt.find(path)
        if el is not None and el.text:
            try:
                return float(el.text)
            except ValueError:
                pass
    return None


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_gpx(file_bytes: bytes) -> GPXSummary:
    """
    Parse a GPX file and return activity metrics and a TSS estimate.

    Args:
        file_bytes: Raw bytes of the .gpx file.

    Returns:
        A GPXSummary with distance, duration, elevation, HR, and TSS.

    Raises:
        ValueError: If the file cannot be parsed or contains no track points.
    """
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Could not read GPX file: {e}")

    # Strip namespace from tag names for easier traversal.
    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # ── Collect all track points ─────────────────────────────────────────────
    trkpts: List[ET.Element] = []
    for elem in root.iter():
        if strip_ns(elem.tag) == "trkpt":
            trkpts.append(elem)

    if len(trkpts) < 2:
        raise ValueError("GPX file contains too few track points to analyse.")

    # ── Activity name ────────────────────────────────────────────────────────
    activity_name = "Cycling activity"
    for elem in root.iter():
        if strip_ns(elem.tag) == "name" and elem.text:
            activity_name = elem.text.strip()
            break

    # ── Parse each track point ───────────────────────────────────────────────
    latitudes:  List[float] = []
    longitudes: List[float] = []
    elevations: List[float] = []
    timestamps: List[datetime] = []
    heart_rates: List[float] = []

    for pt in trkpts:
        try:
            lat = float(pt.get("lat", 0))
            lon = float(pt.get("lon", 0))
        except ValueError:
            continue

        latitudes.append(lat)
        longitudes.append(lon)

        # Elevation
        for child in pt:
            if strip_ns(child.tag) == "ele" and child.text:
                try:
                    elevations.append(float(child.text))
                except ValueError:
                    pass

        # Timestamp
        for child in pt:
            if strip_ns(child.tag) == "time" and child.text:
                try:
                    ts_str = child.text.strip().replace("Z", "+00:00")
                    timestamps.append(datetime.fromisoformat(ts_str))
                except ValueError:
                    pass

        # Heart rate
        hr = _find_hr(pt)
        if hr is not None:
            heart_rates.append(hr)

    # ── Distance ─────────────────────────────────────────────────────────────
    total_distance_km = sum(
        _haversine_km(latitudes[i], longitudes[i],
                      latitudes[i + 1], longitudes[i + 1])
        for i in range(len(latitudes) - 1)
    )

    # ── Duration ─────────────────────────────────────────────────────────────
    if len(timestamps) >= 2:
        duration_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    else:
        # Fallback: estimate 20 km/h average if no timestamps
        duration_seconds = (total_distance_km / 20.0) * 3600

    # ── Elevation gain ───────────────────────────────────────────────────────
    elevation_gain_m = 0.0
    for i in range(len(elevations) - 1):
        delta = elevations[i + 1] - elevations[i]
        if delta > 0:
            elevation_gain_m += delta

    # ── TSS estimate ─────────────────────────────────────────────────────────
    duration_hrs = duration_seconds / 3600.0

    if heart_rates:
        avg_hr = sum(heart_rates) / len(heart_rates)
        intensity = avg_hr / ESTIMATED_MAX_HR
        method = "heart_rate"
    else:
        avg_hr = None
        avg_speed = total_distance_km / duration_hrs if duration_hrs > 0 else 0
        intensity = _speed_intensity(avg_speed)
        method = "speed"

    estimated_tss = round(duration_hrs * (intensity ** 2) * 100, 1)

    return GPXSummary(
        distance_km=round(total_distance_km, 2),
        duration_seconds=round(duration_seconds, 0),
        elevation_gain_m=round(elevation_gain_m, 0),
        avg_hr_bpm=round(avg_hr, 0) if avg_hr else None,
        estimated_tss=estimated_tss,
        method=method,
        activity_name=activity_name,
    )
