"""Trip/chapter clustering by timestamp gaps and GPS distance, with optional geocoding."""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from keepsake._config import TripConfig
from keepsake._logging import get_logger
from keepsake._models import Manifest, PhotoRecord, TripChapter

logger = get_logger(__name__)


class Geocoder(Protocol):
    """Protocol for reverse-geocoding (lat, lon) to a human-readable place name."""

    def reverse(self, lat: float, lon: float) -> str | None: ...


class NominatimGeocoder:
    """Polite OpenStreetMap Nominatim reverse geocoder with local JSON caching."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        user_agent: str = "keepsake-photo-toolkit/0.1",
        min_interval_sec: float = 1.1,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.user_agent = user_agent
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0
        self._cache: dict[str, str | None] = self._load_cache()

    def reverse(self, lat: float, lon: float) -> str | None:
        key = f"{lat:.2f},{lon:.2f}"
        if key in self._cache:
            return self._cache[key]

        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

        params = urllib.parse.urlencode(
            {"format": "jsonv2", "lat": f"{lat:.5f}", "lon": f"{lon:.5f}", "zoom": "10"}
        )
        url = f"https://nominatim.openstreetmap.org/reverse?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        place: str | None = None
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            place = _format_nominatim_place(data)
        except Exception as exc:
            logger.warning("Reverse geocoding failed for (%.2f, %.2f): %s", lat, lon, exc)
            place = None
        finally:
            self._last_call = time.monotonic()

        self._cache[key] = place
        self._save_cache()
        return place

    def _load_cache(self) -> dict[str, str | None]:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates in kilometers."""
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cluster_trips(
    manifest: Manifest,
    config: TripConfig | None = None,
    geocoder: Geocoder | None = None,
) -> list[TripChapter]:
    """Group curated photos into sequential trip chapters and attach to `manifest`."""
    cfg = config or TripConfig()
    active = [p for p in manifest.photos if p.selected_for_mosaic]
    active.sort(key=lambda r: (r.timestamp or "9999", r.path))

    if not active:
        manifest.trips = []
        return []

    if geocoder is None and cfg.geocode:
        geocoder = NominatimGeocoder(cache_path=cfg.geocode_cache_path)

    groups: list[list[PhotoRecord]] = [[active[0]]]
    for photo in active[1:]:
        prev = groups[-1][-1]
        if _starts_new_trip(prev, photo, cfg):
            groups.append([photo])
        else:
            groups[-1].append(photo)

    chapters: list[TripChapter] = []
    for idx, group in enumerate(groups, start=1):
        chapter = _build_chapter(idx, group, cfg, geocoder)
        chapters.append(chapter)
        for photo in group:
            photo.trip_index = idx

    manifest.trips = chapters
    return chapters


def _starts_new_trip(prev: PhotoRecord, curr: PhotoRecord, cfg: TripConfig) -> bool:
    prev_dt = prev.parsed_timestamp
    curr_dt = curr.parsed_timestamp
    if prev_dt is not None and curr_dt is not None:
        gap_hours = abs((curr_dt - prev_dt).total_seconds()) / 3600.0
        if gap_hours > cfg.max_gap_hours:
            return True

    if (
        prev.latitude is not None
        and prev.longitude is not None
        and curr.latitude is not None
        and curr.longitude is not None
    ):
        dist = haversine_km(prev.latitude, prev.longitude, curr.latitude, curr.longitude)
        if dist > cfg.max_distance_km:
            return True

    return False


def _build_chapter(
    index: int,
    group: list[PhotoRecord],
    cfg: TripConfig,
    geocoder: Geocoder | None,
) -> TripChapter:
    timestamps = [p.parsed_timestamp for p in group if p.parsed_timestamp is not None]
    start_dt = min(timestamps) if timestamps else None
    end_dt = max(timestamps) if timestamps else None

    coords = [
        (p.latitude, p.longitude)
        for p in group
        if p.latitude is not None and p.longitude is not None
    ]
    center_lat = round(sum(c[0] for c in coords) / len(coords), 5) if coords else None
    center_lon = round(sum(c[1] for c in coords) / len(coords), 5) if coords else None

    location_name: str | None = None
    if geocoder is not None and center_lat is not None and center_lon is not None:
        location_name = geocoder.reverse(center_lat, center_lon)

    date_label = _format_date_range(start_dt, end_dt)
    if location_name and date_label:
        title = f"{location_name} — {date_label}"
    elif location_name:
        title = location_name
    elif date_label:
        title = f"{cfg.default_chapter_prefix} {index} — {date_label}"
    else:
        title = f"{cfg.default_chapter_prefix} {index}"

    return TripChapter(
        index=index,
        title=title,
        start_date=start_dt.date().isoformat() if start_dt else None,
        end_date=end_dt.date().isoformat() if end_dt else None,
        center_lat=center_lat,
        center_lon=center_lon,
        location_name=location_name,
        photo_paths=[p.path for p in group],
    )


def _format_date_range(start_dt, end_dt) -> str:
    if start_dt is None:
        return ""
    if end_dt is None or (start_dt.year == end_dt.year and start_dt.month == end_dt.month):
        return start_dt.strftime("%B %Y")
    if start_dt.year == end_dt.year:
        return f"{start_dt.strftime('%b')}–{end_dt.strftime('%b %Y')}"
    return f"{start_dt.strftime('%b %Y')}–{end_dt.strftime('%b %Y')}"


def _format_nominatim_place(data: dict) -> str | None:
    addr = data.get("address", {})
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("state")
    )
    country = addr.get("country")
    if city and country:
        return f"{city}, {country}"
    return city or country or data.get("name")
