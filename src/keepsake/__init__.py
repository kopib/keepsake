"""keepsake — reusable photo curation, trip clustering, video, and mosaic toolkit."""

from keepsake._config import (
    AudioClipConfig,
    IngestConfig,
    MosaicConfig,
    ProjectConfig,
    ProjectPathsConfig,
    TripConfig,
    VideoConfig,
)
from keepsake._curate import scan_photos, select_video_photos
from keepsake._logging import configure_logging, get_logger
from keepsake._models import Manifest, PhotoRecord, TripChapter
from keepsake._mosaic import (
    MosaicResult,
    build_mosaic,
    render_mosaic_puzzle_finale,
    render_mosaic_zoom_reveal,
)
from keepsake._pacing import PacingPlan, PlannedClip, plan_video_pacing
from keepsake._trips import cluster_trips
from keepsake._video import VideoResult, render_keepsake_video

__all__ = [
    "AudioClipConfig",
    "IngestConfig",
    "Manifest",
    "MosaicConfig",
    "MosaicResult",
    "PacingPlan",
    "PhotoRecord",
    "PlannedClip",
    "ProjectConfig",
    "ProjectPathsConfig",
    "TripChapter",
    "TripConfig",
    "VideoConfig",
    "VideoResult",
    "build_mosaic",
    "cluster_trips",
    "configure_logging",
    "get_logger",
    "plan_video_pacing",
    "render_keepsake_video",
    "render_mosaic_puzzle_finale",
    "render_mosaic_zoom_reveal",
    "scan_photos",
    "select_video_photos",
]
