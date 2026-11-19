"""Photo folder scanning, burst deduplication, sharpness filtering, and video selection."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from keepsake._config import IngestConfig
from keepsake._image_io import ensure_heif_support
from keepsake._metadata import extract_photo_record, hamming_distance
from keepsake._models import Manifest, PhotoRecord, TripChapter


def discover_image_files(root: str | Path, config: IngestConfig) -> list[Path]:
    """Find supported image files under `root`, sorted deterministically."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Photo directory not found: {root_path}")

    exts = {ext.lower() for ext in config.extensions}
    iterator = root_path.rglob("*") if config.recursive else root_path.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    ]
    return sorted(files)


def scan_photos(root: str | Path, config: IngestConfig | None = None) -> Manifest:
    """Scan a directory of photos, deduplicate bursts, and flag low-sharpness shots."""
    cfg = config or IngestConfig()
    files = discover_image_files(root, cfg)

    workers = cfg.scan_workers if cfg.scan_workers > 0 else min(8, os.cpu_count() or 4)
    if workers > 1 and len(files) > 1:
        ensure_heif_support()

        def _extract(path: Path) -> PhotoRecord:
            return extract_photo_record(
                path,
                camera_time_offsets_hours=cfg.camera_time_offsets_hours,
            )

        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
            records = list(pool.map(_extract, files))
    else:
        records = [
            extract_photo_record(
                path,
                camera_time_offsets_hours=cfg.camera_time_offsets_hours,
            )
            for path in files
        ]

    records.sort(key=lambda r: (r.timestamp or "9999", r.path))
    apply_curation(records, cfg)
    return Manifest(photos=records, metadata={"source_dir": str(Path(root).resolve())})


def apply_curation(records: list[PhotoRecord], config: IngestConfig) -> None:
    """Mark blurry images and near-duplicate bursts in-place (keeping sharpest)."""
    for rec in records:
        is_sharp = rec.sharpness >= config.min_sharpness
        rec.selected_for_mosaic = is_sharp
        rec.selected_for_video = is_sharp
        rec.burst_group_id = None

    if config.dedup_hash_distance < 0 or len(records) < 2:
        return

    active = [r for r in records if r.selected_for_mosaic]
    # Greedy burst grouping among nearby records (sorted by timestamp)
    kept: list[PhotoRecord] = []
    next_burst_id = 1
    for candidate in active:
        duplicate_idx = _find_similar_recent(
            candidate, kept, max_distance=config.dedup_hash_distance
        )
        if duplicate_idx is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_idx]
        # Assign a shared burst_group_id so dynamic video pacing can optionally use rapid frames
        if existing.burst_group_id is None:
            existing.burst_group_id = next_burst_id
            next_burst_id += 1
        candidate.burst_group_id = existing.burst_group_id

        if candidate.sharpness > existing.sharpness:
            existing.selected_for_mosaic = False
            existing.selected_for_video = False
            kept[duplicate_idx] = candidate
        else:
            candidate.selected_for_mosaic = False
            candidate.selected_for_video = False


def select_video_photos(
    manifest: Manifest,
    max_photos: int,
) -> list[PhotoRecord]:
    """Select up to `max_photos` evenly across trips (or timeline) for video pacing."""
    eligible = [p for p in manifest.photos if p.selected_for_mosaic]
    if max_photos <= 0 or len(eligible) <= max_photos:
        for p in manifest.photos:
            p.selected_for_video = p.selected_for_mosaic
        return eligible

    selected_paths: set[str] = set()
    if manifest.trips:
        selected_paths = _allocate_across_trips(manifest.trips, eligible, max_photos)
    else:
        selected_paths = {p.path for p in _sample_evenly(eligible, max_photos)}

    for p in manifest.photos:
        p.selected_for_video = p.path in selected_paths
    return [p for p in manifest.photos if p.selected_for_video]


def _find_similar_recent(
    candidate: PhotoRecord,
    kept: list[PhotoRecord],
    *,
    max_distance: int,
    window: int = 25,
) -> int | None:
    start = max(0, len(kept) - window)
    for idx in range(len(kept) - 1, start - 1, -1):
        if hamming_distance(candidate.dhash, kept[idx].dhash) <= max_distance:
            return idx
    return None


def _allocate_across_trips(
    trips: list[TripChapter],
    eligible: list[PhotoRecord],
    max_photos: int,
) -> set[str]:
    by_path = {p.path: p for p in eligible}
    trip_buckets = [
        [by_path[path] for path in trip.photo_paths if path in by_path] for trip in trips
    ]
    trip_buckets = [b for b in trip_buckets if b]
    if not trip_buckets:
        return {p.path for p in _sample_evenly(eligible, max_photos)}

    total_eligible = sum(len(b) for b in trip_buckets)
    # Proportional allocation with at least 1 photo per trip (when max_photos >= num_trips)
    quotas = [max(1, round(max_photos * len(b) / total_eligible)) for b in trip_buckets]
    # Adjust quotas down or up so sum matches max_photos
    while sum(quotas) > max_photos:
        largest = max(range(len(quotas)), key=lambda i: quotas[i])
        if quotas[largest] <= 1:
            break
        quotas[largest] -= 1
    while sum(quotas) < max_photos:
        candidates = [i for i, b in enumerate(trip_buckets) if quotas[i] < len(b)]
        if not candidates:
            break
        best = max(candidates, key=lambda i: len(trip_buckets[i]) - quotas[i])
        quotas[best] += 1

    chosen: set[str] = set()
    for bucket, quota in zip(trip_buckets, quotas, strict=True):
        for photo in _sample_evenly(bucket, quota):
            chosen.add(photo.path)
    return chosen


def _sample_evenly(items: list[PhotoRecord], count: int) -> list[PhotoRecord]:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [max(items, key=lambda r: (r.sharpness, not r.is_portrait))]

    # Divide bucket into `count` chronological windows and pick the strongest shot in each
    chosen: list[PhotoRecord] = []
    n = len(items)
    for slot in range(count):
        start = round(slot * n / count)
        end = max(start + 1, round((slot + 1) * n / count))
        window = items[start:end]
        is_anchor = slot == 0 or slot == count - 1
        best = max(
            window,
            key=lambda r: (
                r.sharpness * (1.15 if not r.is_portrait else 1.0),
                1 if is_anchor else 0,
            ),
        )
        chosen.append(best)
    return chosen
