"""Configuration dataclasses, print/video presets, and TOML loader."""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from keepsake._audio import (
    AudioTrackSpec,
    discover_audio_files,
    expand_audio_inputs,
    parse_timestamp_sec,
    split_audio_shorthand,
)

PAPER_SIZES_CM: dict[str, tuple[float, float]] = {
    "A4": (21.0, 29.7),
    "A3": (29.7, 42.0),
    "A2": (42.0, 59.4),
    "30X40": (30.0, 40.0),
    "50X70": (50.0, 70.0),
}

VIDEO_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "4K": (3840, 2160),
    "1080P": (1920, 1080),
    "720P": (1280, 720),
}

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".orf",
)


@dataclass(slots=True)
class IngestConfig:
    """Settings for scanning, time-offset calibration, and deduplication."""

    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    camera_time_offsets_hours: dict[str, float] = field(default_factory=dict)
    dedup_hash_distance: int = 6
    min_sharpness: float = 15.0
    max_video_photos: int = 75
    recursive: bool = True
    scan_workers: int = 0  # 0 = auto (up to 8 threads); 1 = sequential


@dataclass(slots=True)
class TripConfig:
    """Settings for clustering photos into chronological/geographical trips."""

    max_gap_hours: float = 48.0
    max_distance_km: float = 80.0
    geocode: bool = False
    geocode_cache_path: str | None = None
    default_chapter_prefix: str = "Chapter"


@dataclass(slots=True)
class VideoConfig:
    """Settings for Ken Burns wallpaper video generation via ffmpeg."""

    resolution: str = "4K"
    width: int | None = None
    height: int | None = None
    fps: int = 30
    target_duration_sec: float = 360.0
    seconds_per_photo: float | None = None
    pacing_mode: Literal["uniform", "dynamic"] = "dynamic"
    target_avg_clip_sec: float = 0.85
    min_clip_sec: float = 0.20
    max_clip_sec: float = 2.40
    dynamic_fill_from_curated: bool = True
    beat_sync: bool = True
    motion_sync: bool = True
    camera_motion: Literal["zoom", "subtle_pan", "none"] = "zoom"
    transition_sec: float = 0.8
    portrait_mode: Literal["blurred_bg", "pad", "crop"] = "blurred_bg"
    include_title_cards: bool = True
    title_card_mode: Literal["year", "chapter", "none"] = "year"
    year_card_prefix: str = "Year "
    title_card_sec: float = 2.4
    closing_quote: str | None = None
    closing_quote_sec: float = 3.5
    mosaic_finale_sec: float = 12.0
    mosaic_finale_style: Literal["puzzle", "zoom"] = "puzzle"
    mosaic_puzzle_pattern: Literal["fly_in", "center_out", "spiral"] = "fly_in"
    mosaic_focal_x: float = 0.5
    mosaic_focal_y: float = 0.5
    codec: Literal["libx264", "libx265"] = "libx264"
    crf: int = 18
    preset: str = "medium"
    audio_fade_in_sec: float = 1.0
    audio_fade_out_sec: float = 3.0
    audio_crossfade_sec: float = 2.0
    audio_normalize: bool = True
    auto_climax: bool = False
    climax_duration_sec: float = 40.0
    zoom_strength: float = 0.08
    batch_size: int = 8

    def resolved_dimensions(self) -> tuple[int, int]:
        if self.width is not None and self.height is not None:
            return self.width, self.height
        key = self.resolution.upper()
        if key not in VIDEO_RESOLUTIONS:
            raise ValueError(
                f"Unknown video resolution '{self.resolution}'. "
                f"Choose from {sorted(VIDEO_RESOLUTIONS)} or set width/height."
            )
        return VIDEO_RESOLUTIONS[key]


@dataclass(slots=True)
class MosaicConfig:
    """Settings for print-ready photomosaic generation."""

    paper_size: str = "A4"
    width_cm: float | None = None
    height_cm: float | None = None
    dpi: int = 300
    cols: int = 36
    subgrid: int = 3
    max_reuse: int = 8
    min_distance: int = 3
    overlay_alpha: float = 0.14
    auto_orient: bool = True
    crop_mode: Literal["center", "fit"] = "center"
    seed: int = 42

    def resolved_canvas_px(self, target_width: int, target_height: int) -> tuple[int, int]:
        """Compute exact pixel canvas dimensions matching paper size and orientation."""
        if self.width_cm is not None and self.height_cm is not None:
            w_cm, h_cm = self.width_cm, self.height_cm
        else:
            key = self.paper_size.upper()
            if key not in PAPER_SIZES_CM:
                raise ValueError(
                    f"Unknown paper size '{self.paper_size}'. "
                    f"Choose from {sorted(PAPER_SIZES_CM)} or set width_cm/height_cm."
                )
            w_cm, h_cm = PAPER_SIZES_CM[key]

        if self.auto_orient:
            target_landscape = target_width > target_height
            paper_landscape = w_cm > h_cm
            if target_landscape != paper_landscape:
                w_cm, h_cm = h_cm, w_cm

        px_w = round((w_cm / 2.54) * self.dpi)
        px_h = round((h_cm / 2.54) * self.dpi)
        return px_w, px_h


@dataclass(slots=True)
class AudioClipConfig:
    """Structured audio track entry with optional start/end climax window."""

    path: str
    start: str | float | None = None
    end: str | float | None = None
    full: bool = False


@dataclass(slots=True)
class ProjectPathsConfig:
    """Configurable input/output paths so runs can be driven solely from a TOML file."""

    photos_dir: str | None = None
    target_image: str | None = None
    out_dir: str = "output"
    manifest: str | None = None
    mosaic_image: str | None = None
    audio_dir: str | None = None
    audio_tracks: tuple[str, ...] = ()
    audio_clips: tuple[AudioClipConfig, ...] = ()

    def resolve_relative_to(self, base_dir: Path) -> None:
        """Resolve non-empty relative path fields against a config file's directory."""
        if self.photos_dir:
            self.photos_dir = str(_resolve_path(self.photos_dir, base_dir))
        if self.target_image:
            self.target_image = str(_resolve_path(self.target_image, base_dir))
        if self.out_dir:
            self.out_dir = str(_resolve_path(self.out_dir, base_dir))
        if self.manifest:
            self.manifest = str(_resolve_path(self.manifest, base_dir))
        if self.mosaic_image:
            self.mosaic_image = str(_resolve_path(self.mosaic_image, base_dir))
        if self.audio_dir:
            self.audio_dir = str(_resolve_path(self.audio_dir, base_dir))
        if self.audio_tracks:
            resolved_tracks: list[str] = []
            for track in self.audio_tracks:
                raw_path, suffix = split_audio_shorthand(track)
                full_path = str(_resolve_path(raw_path, base_dir))
                resolved_tracks.append(f"{full_path}@{suffix}" if suffix else full_path)
            self.audio_tracks = tuple(resolved_tracks)
        for clip in self.audio_clips:
            raw_clip = Path(clip.path).expanduser()
            if not raw_clip.is_absolute() and self.audio_dir:
                candidate_in_dir = (Path(self.audio_dir) / raw_clip).resolve()
                candidate_in_base = (base_dir / raw_clip).resolve()
                if not candidate_in_base.exists() and candidate_in_dir.exists():
                    clip.path = str(candidate_in_dir)
                    continue
            clip.path = str(_resolve_path(clip.path, base_dir))

    def resolved_manifest_path(self) -> Path:
        if self.manifest:
            return Path(self.manifest)
        return Path(self.out_dir) / "manifest.json"

    def resolved_mosaic_path(self) -> Path:
        if self.mosaic_image:
            return Path(self.mosaic_image)
        return Path(self.out_dir) / "mosaic.jpg"

    def resolved_audio_specs(self) -> list[AudioTrackSpec]:
        """Return structured AudioTrackSpec list from `audio_dir`, `audio_clips`, or `audio_tracks`."""
        clip_specs = [
            AudioTrackSpec(
                path=Path(clip.path),
                start_sec=parse_timestamp_sec(clip.start),
                end_sec=parse_timestamp_sec(clip.end),
                force_full=clip.full,
            )
            for clip in self.audio_clips
        ]
        if self.audio_dir and clip_specs:
            specs = [AudioTrackSpec(path=p) for p in discover_audio_files(self.audio_dir)]
            by_path = {s.path.resolve(): idx for idx, s in enumerate(specs)}
            for override in clip_specs:
                key = override.path.resolve()
                if key in by_path:
                    specs[by_path[key]] = override
                else:
                    specs.append(override)
            return specs
        if clip_specs:
            return clip_specs
        if self.audio_tracks:
            return expand_audio_inputs(list(self.audio_tracks))
        if self.audio_dir:
            return [AudioTrackSpec(path=p) for p in discover_audio_files(self.audio_dir)]
        return []


@dataclass(slots=True)
class ProjectConfig:
    """Top-level configuration combining all pipeline stages."""

    project: ProjectPathsConfig = field(default_factory=ProjectPathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    trips: TripConfig = field(default_factory=TripConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ProjectConfig:
        toml_path = Path(path).resolve()
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        known_sections = {"project", "ingest", "trips", "video", "mosaic"}
        unknown_sections = sorted(set(raw) - known_sections)
        if unknown_sections:
            warnings.warn(
                f"Unknown TOML section(s) in {path}: {', '.join(unknown_sections)}",
                UserWarning,
                stacklevel=2,
            )
        project_cfg = _build_section(
            ProjectPathsConfig, raw.get("project", {}), section_name="project"
        )
        project_cfg.resolve_relative_to(toml_path.parent)
        return cls(
            project=project_cfg,
            ingest=_build_section(IngestConfig, raw.get("ingest", {}), section_name="ingest"),
            trips=_build_section(TripConfig, raw.get("trips", {}), section_name="trips"),
            video=_build_section(VideoConfig, raw.get("video", {}), section_name="video"),
            mosaic=_build_section(MosaicConfig, raw.get("mosaic", {}), section_name="mosaic"),
        )


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def _build_section(
    dc_type: type[Any],
    data: dict[str, Any],
    *,
    section_name: str = "",
) -> Any:
    valid_names = {f.name for f in fields(dc_type)}
    unknown_keys = sorted(set(data) - valid_names)
    if unknown_keys:
        prefix = f"[{section_name}] " if section_name else ""
        warnings.warn(
            f"Unknown config key(s) in {prefix}{dc_type.__name__}: {', '.join(unknown_keys)}",
            UserWarning,
            stacklevel=3,
        )
    filtered = {k: v for k, v in data.items() if k in valid_names}
    for tuple_key in ("extensions", "audio_tracks"):
        if tuple_key in filtered and isinstance(filtered[tuple_key], list):
            filtered[tuple_key] = tuple(filtered[tuple_key])
    if "audio_clips" in filtered and isinstance(filtered["audio_clips"], list):
        filtered["audio_clips"] = tuple(
            _build_section(AudioClipConfig, item, section_name="project.audio_clips")
            for item in filtered["audio_clips"]
            if isinstance(item, dict)
        )
    return dc_type(**filtered)
