"""Data models and JSON manifest serialization for keepsake."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PhotoRecord:
    """Metadata and quality metrics for a single scanned image."""

    path: str
    timestamp: str | None
    timestamp_source: str
    latitude: float | None
    longitude: float | None
    camera_make: str | None
    camera_model: str | None
    width: int
    height: int
    sharpness: float
    dhash: str
    trip_index: int | None = None
    selected_for_video: bool = True
    selected_for_mosaic: bool = True
    burst_group_id: int | None = None
    video_role: str | None = None
    duration_sec: float | None = None
    transition_sec: float | None = None
    pacing_reason: str | None = None
    audio_energy: float | None = None

    @property
    def parsed_timestamp(self) -> datetime | None:
        if not self.timestamp:
            return None
        return datetime.fromisoformat(self.timestamp)

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


@dataclass(slots=True)
class TripChapter:
    """A cluster of photos grouped by time and/or location."""

    index: int
    title: str
    start_date: str | None
    end_date: str | None
    center_lat: float | None
    center_lon: float | None
    location_name: str | None
    photo_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Manifest:
    """Serializable pipeline state connecting scan, trip grouping, video, and mosaic."""

    photos: list[PhotoRecord] = field(default_factory=list)
    trips: list[TripChapter] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "photos": [asdict(p) for p in self.photos],
            "trips": [asdict(t) for t in self.trips],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        photos = [PhotoRecord(**item) for item in raw.get("photos", [])]
        trips = [TripChapter(**item) for item in raw.get("trips", [])]
        return cls(
            photos=photos,
            trips=trips,
            metadata=raw.get("metadata", {}),
        )
