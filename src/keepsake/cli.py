"""Command-line interface for the keepsake photo, video, and mosaic pipeline."""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

from keepsake._audio import (
    AudioTrackSpec,
    analyze_audio_dynamics,
    estimate_segments_duration,
    expand_audio_inputs,
    format_timestamp,
    mix_audio_segments,
    resolve_audio_segments,
)
from keepsake._config import MosaicConfig, ProjectConfig, VideoConfig
from keepsake._curate import scan_photos, select_video_photos
from keepsake._logging import configure_logging, get_logger
from keepsake._models import Manifest
from keepsake._mosaic import build_mosaic
from keepsake._pacing import PacingPlan, plan_video_pacing
from keepsake._trips import cluster_trips
from keepsake._video import _estimate_title_card_count, render_keepsake_video

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keepsake",
        description="Curate photos, cluster trips, render Ken Burns videos, and build print mosaics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML configuration file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Scan a photo directory and build manifest.json.")
    p_scan.add_argument(
        "photos_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Folder containing photos (or set [project].photos_dir in config).",
    )
    p_scan.add_argument(
        "-o",
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest JSON (default: [project].manifest or manifest.json).",
    )
    p_scan.add_argument("--geocode", action="store_true", help="Reverse-geocode trip centroids.")
    p_scan.add_argument("--max-video-photos", type=int, default=None, help="Max photos for video.")

    # mosaic
    p_mos = sub.add_parser("mosaic", help="Build a print-ready photomosaic portrait.")
    p_mos.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=None,
        help="Path to manifest.json (or set via [project] in config).",
    )
    p_mos.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=None,
        help="Target portrait image (or set [project].target_image in config).",
    )
    p_mos.add_argument("-o", "--output", type=Path, default=None, help="Output image.")
    p_mos.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Paper preset: A4, A3, A2, 30x40, 50x70.",
    )
    p_mos.add_argument("--dpi", type=int, default=None, help="Output DPI (default: 300).")
    p_mos.add_argument("--cols", type=int, default=None, help="Tile columns across width.")
    p_mos.add_argument(
        "--color-correction",
        type=float,
        default=None,
        help="Per-tile CIE Lab color/luminance matching strength (0.0-1.0).",
    )
    p_mos.add_argument("--overlay-alpha", type=float, default=None, help="Target blend (0.0-1.0).")

    # video
    p_vid = sub.add_parser("video", help="Render a wallpaper-style Ken Burns video.")
    p_vid.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=None,
        help="Path to manifest.json (or set via [project] in config).",
    )
    p_vid.add_argument("-o", "--output", type=Path, default=None, help="Output MP4.")
    p_vid.add_argument("--resolution", type=str, default=None, help="4K, 1080P, or 720P.")
    p_vid.add_argument(
        "--duration", type=float, default=None, help="Target video duration in seconds."
    )
    p_vid.add_argument(
        "--audio",
        type=str,
        nargs="*",
        default=None,
        help="Audio track(s), optionally with @MM:SS-MM:SS climax window.",
    )
    p_vid.add_argument(
        "--auto-climax", action="store_true", help="Auto-detect peak climax per song."
    )
    p_vid.add_argument(
        "--climax-sec",
        type=float,
        default=None,
        help="Climax window length in seconds.",
    )
    p_vid.add_argument(
        "--mosaic-finale",
        type=Path,
        default=None,
        help="Optional mosaic image for zoom-out finale (auto-uses <out_dir>/mosaic.jpg with --config).",
    )
    p_vid.add_argument(
        "--no-mosaic-finale",
        action="store_true",
        help="Disable mosaic zoom-out finale.",
    )
    p_vid.add_argument("--no-title-cards", action="store_true", help="Disable chapter title cards.")
    p_vid.add_argument(
        "--pacing", choices=("dynamic", "uniform"), default=None, help="Pacing mode."
    )
    p_vid.add_argument(
        "--replan",
        action="store_true",
        help="Recompute variable pacing even if manifest has durations.",
    )

    # plan (compute & save variable pacing to manifest without rendering)
    p_plan = sub.add_parser(
        "plan", help="Plan context-aware variable photo durations into manifest.json."
    )
    p_plan.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=None,
        help="Path to manifest.json (or set via [project] in config).",
    )
    p_plan.add_argument(
        "--audio",
        type=str,
        nargs="*",
        default=None,
        help="Audio track(s), optionally with @MM:SS-MM:SS climax window.",
    )
    p_plan.add_argument(
        "--auto-climax", action="store_true", help="Auto-detect peak climax per song."
    )
    p_plan.add_argument(
        "--climax-sec",
        type=float,
        default=None,
        help="Climax window length in seconds.",
    )
    p_plan.add_argument(
        "--no-title-cards",
        action="store_true",
        help="Exclude title card overhead from plan.",
    )
    p_plan.add_argument(
        "--mosaic-finale",
        type=Path,
        default=None,
        help="Account for mosaic finale duration.",
    )
    p_plan.add_argument(
        "--no-mosaic-finale",
        action="store_true",
        help="Exclude mosaic finale from plan.",
    )

    # audio (inspect / preview climax mix)
    p_aud = sub.add_parser(
        "audio",
        help="Detect/slice song climaxes, inspect beats, and optionally export mix.",
    )
    p_aud.add_argument(
        "tracks",
        type=str,
        nargs="*",
        help="Audio track(s), optionally with @MM:SS-MM:SS (defaults to [project] config).",
    )
    p_aud.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output preview audio (.m4a/.wav); defaults to <out_dir>/mix_preview.m4a with --config.",
    )
    p_aud.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect climax windows and beats without writing an audio file.",
    )
    p_aud.add_argument(
        "--auto-climax", action="store_true", help="Auto-detect peak climax per song."
    )
    p_aud.add_argument(
        "--climax-sec",
        type=float,
        default=None,
        help="Climax window length in seconds.",
    )

    # run (all-in-one)
    p_run = sub.add_parser("run", help="Run scan + trips + mosaic + video end-to-end.")
    p_run.add_argument(
        "photos_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Folder containing photos (or set [project].photos_dir in config).",
    )
    p_run.add_argument("--target", type=Path, default=None, help="Target portrait for mosaic.")
    p_run.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    p_run.add_argument(
        "--audio",
        type=str,
        nargs="*",
        default=None,
        help="Audio track(s), optionally with @MM:SS-MM:SS climax window.",
    )
    p_run.add_argument(
        "--auto-climax", action="store_true", help="Auto-detect peak climax per song."
    )
    p_run.add_argument(
        "--climax-sec",
        type=float,
        default=None,
        help="Climax window length in seconds.",
    )
    p_run.add_argument("--geocode", action="store_true", help="Reverse-geocode trip locations.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    prev_showwarning = warnings.showwarning

    def _log_warning(message, category, filename, lineno, file=None, line=None) -> None:
        logger.warning("%s", message)

    warnings.showwarning = _log_warning
    try:
        return _dispatch(parser, args)
    finally:
        warnings.showwarning = prev_showwarning


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    cfg = ProjectConfig.from_toml(args.config) if args.config else ProjectConfig()

    if args.command == "scan":
        photos_dir = args.photos_dir or (
            Path(cfg.project.photos_dir) if cfg.project.photos_dir else None
        )
        if photos_dir is None:
            parser.error(
                "photos_dir is required (pass as argument or set [project].photos_dir in config)."
            )
        manifest_out = args.manifest or (
            cfg.project.resolved_manifest_path() if args.config else Path("manifest.json")
        )
        if args.geocode:
            cfg.trips.geocode = True
        if args.max_video_photos is not None:
            cfg.ingest.max_video_photos = args.max_video_photos
        manifest = _run_scan_and_trips(photos_dir, cfg)
        manifest.save(manifest_out)
        _print_scan_summary(manifest, manifest_out)
        return 0

    if args.command == "mosaic":
        if args.manifest is None and not args.config:
            parser.error("manifest is required (pass as argument or provide --config).")
        manifest_path = args.manifest or cfg.project.resolved_manifest_path()
        target_path = args.target or (
            Path(cfg.project.target_image) if cfg.project.target_image else None
        )
        if target_path is None:
            parser.error(
                "target is required (pass as argument or set [project].target_image in config)."
            )
        out_path = args.output or (
            cfg.project.resolved_mosaic_path() if args.config else Path("mosaic.jpg")
        )
        manifest = Manifest.load(manifest_path)
        m_cfg = _override_mosaic_config(cfg.mosaic, args)
        tile_paths = [p.path for p in manifest.photos if p.selected_for_mosaic]
        res = build_mosaic(target_path, tile_paths, out_path, config=m_cfg)
        logger.info(
            "Mosaic saved to %s (%dx%dpx @ %d DPI, %dx%d grid, %d unique tiles)",
            res.output_path,
            res.width_px,
            res.height_px,
            res.dpi,
            res.cols,
            res.rows,
            res.unique_tiles_used,
        )
        return 0

    if args.command == "audio":
        audio_specs = _resolve_cli_audio_specs(args.tracks or None, cfg)
        if not audio_specs:
            parser.error(
                "Provide audio tracks as arguments or set [project].audio_tracks / audio_clips."
            )
        missing = [str(s.path) for s in audio_specs if not s.path.exists()]
        if missing:
            parser.error(f"Audio file(s) not found: {', '.join(missing)}")
        auto_climax = cfg.video.auto_climax or args.auto_climax
        climax_sec = (
            args.climax_sec if args.climax_sec is not None else cfg.video.climax_duration_sec
        )
        segments = resolve_audio_segments(
            audio_specs,
            auto_climax=auto_climax,
            climax_duration_sec=climax_sec,
            crossfade_sec=cfg.video.audio_crossfade_sec,
        )
        total_dur = estimate_segments_duration(
            segments, crossfade_sec=cfg.video.audio_crossfade_sec
        )
        for idx, seg in enumerate(segments, start=1):
            tag = "auto-climax" if seg.auto_detected else "window"
            peak = f", peak @ {format_timestamp(seg.peak_sec)}" if seg.peak_sec is not None else ""
            logger.info(
                "Track %d: %s -> [%s-%s] (%.1fs, %s%s)",
                idx,
                seg.path.name,
                format_timestamp(seg.start_sec),
                format_timestamp(seg.end_sec),
                seg.duration_sec,
                tag,
                peak,
            )

        out_path = (
            None
            if args.dry_run
            else (
                args.output
                or (Path(cfg.project.out_dir) / "mix_preview.m4a" if args.config else None)
            )
        )
        if out_path is not None:
            mix_audio_segments(
                segments,
                out_path,
                crossfade_sec=cfg.video.audio_crossfade_sec,
                fade_in_sec=cfg.video.audio_fade_in_sec,
                fade_out_sec=cfg.video.audio_fade_out_sec,
                normalize=cfg.video.audio_normalize,
            )
            timeline = analyze_audio_dynamics(out_path, segments=segments)
            logger.info(
                "Mixed preview saved to %s (%.1fs, %d detected beats/onsets)",
                out_path,
                total_dur,
                len(timeline.beat_times),
            )
        else:
            with tempfile.TemporaryDirectory(prefix="keepsake_aud_") as tmp_str:
                tmp_wav = Path(tmp_str) / "mix.wav"
                mix_audio_segments(
                    segments,
                    tmp_wav,
                    crossfade_sec=cfg.video.audio_crossfade_sec,
                    fade_in_sec=cfg.video.audio_fade_in_sec,
                    fade_out_sec=cfg.video.audio_fade_out_sec,
                    normalize=cfg.video.audio_normalize,
                )
                timeline = analyze_audio_dynamics(tmp_wav, segments=segments)
            logger.info(
                "Estimated mix: %.1fs, %d detected beats/onsets",
                total_dur,
                len(timeline.beat_times),
            )
        return 0

    if args.command == "plan":
        if args.manifest is None and not args.config:
            parser.error("manifest is required (pass as argument or provide --config).")
        manifest_path = args.manifest or cfg.project.resolved_manifest_path()
        manifest = Manifest.load(manifest_path)
        v_cfg = cfg.video
        if args.no_title_cards:
            v_cfg.include_title_cards = False
        if args.auto_climax:
            v_cfg.auto_climax = True
        if args.climax_sec is not None:
            v_cfg.climax_duration_sec = args.climax_sec

        audio_specs = [s for s in _resolve_cli_audio_specs(args.audio, cfg) if s.path.exists()]
        segments = (
            resolve_audio_segments(
                audio_specs,
                auto_climax=v_cfg.auto_climax,
                climax_duration_sec=v_cfg.climax_duration_sec,
                crossfade_sec=v_cfg.audio_crossfade_sec,
            )
            if audio_specs
            else []
        )
        target_total = (
            estimate_segments_duration(segments, crossfade_sec=v_cfg.audio_crossfade_sec)
            if segments
            else v_cfg.target_duration_sec
        )
        audio_timeline = None
        if segments:
            with tempfile.TemporaryDirectory(prefix="keepsake_plan_") as tmp_str:
                tmp_wav = Path(tmp_str) / "mix.wav"
                mix_audio_segments(
                    segments,
                    tmp_wav,
                    crossfade_sec=v_cfg.audio_crossfade_sec,
                    fade_in_sec=v_cfg.audio_fade_in_sec,
                    fade_out_sec=0.0,
                    normalize=v_cfg.audio_normalize,
                )
                audio_timeline = analyze_audio_dynamics(tmp_wav, segments=segments)

        effective_mode = v_cfg.title_card_mode if v_cfg.include_title_cards else "none"
        num_cards = _estimate_title_card_count(manifest.photos, manifest.trips, mode=effective_mode)
        card_overhead = num_cards * max(0.0, v_cfg.title_card_sec - v_cfg.transition_sec)
        has_quote = (
            bool(v_cfg.closing_quote and v_cfg.closing_quote.strip())
            and v_cfg.closing_quote_sec > 0
        )
        quote_overhead = (
            max(0.0, v_cfg.closing_quote_sec - v_cfg.transition_sec) if has_quote else 0.0
        )
        mosaic_finale = _resolve_mosaic_finale(args, cfg)
        has_finale = mosaic_finale is not None and v_cfg.mosaic_finale_sec > 0
        finale_overhead = (
            max(0.0, v_cfg.mosaic_finale_sec - v_cfg.transition_sec) if has_finale else 0.0
        )
        photo_span = max(2.0, target_total - card_overhead - quote_overhead - finale_overhead)
        plan = plan_video_pacing(
            manifest,
            photo_span,
            v_cfg,
            timeline=audio_timeline,
            persist_to_manifest=True,
        )
        manifest.save(manifest_path)
        _print_pacing_summary(plan, manifest_path)
        return 0

    if args.command == "video":
        if args.manifest is None and not args.config:
            parser.error("manifest is required (pass as argument or provide --config).")
        manifest_path = args.manifest or cfg.project.resolved_manifest_path()
        out_path = args.output or (
            Path(cfg.project.out_dir) / "keepsake.mp4" if args.config else Path("keepsake.mp4")
        )
        audio_specs = _resolve_cli_audio_specs(args.audio, cfg)
        manifest = Manifest.load(manifest_path)
        v_cfg = _override_video_config(cfg.video, args)
        if args.replan:
            for rec in manifest.photos:
                rec.duration_sec = None
                rec.transition_sec = None
        mosaic_finale = _resolve_mosaic_finale(args, cfg)
        res = render_keepsake_video(
            manifest,
            out_path,
            config=v_cfg,
            audio_specs=audio_specs or None,
            mosaic_image_path=mosaic_finale,
            mosaic_cols=cfg.mosaic.cols,
        )
        manifest.save(manifest_path)
        logger.info(
            "Video saved to %s (%.1fs, %dx%d@%dfps, %d photos)",
            res.output_path,
            res.duration_sec,
            res.width,
            res.height,
            res.fps,
            res.photo_count,
        )
        if res.pacing_plan is not None:
            _print_pacing_summary(res.pacing_plan)
        return 0

    if args.command == "run":
        photos_dir = args.photos_dir or (
            Path(cfg.project.photos_dir) if cfg.project.photos_dir else None
        )
        if photos_dir is None:
            parser.error(
                "photos_dir is required (pass as argument or set [project].photos_dir in config)."
            )
        target_path = args.target or (
            Path(cfg.project.target_image) if cfg.project.target_image else None
        )
        out_dir: Path = args.out_dir or Path(cfg.project.out_dir)
        audio_specs = _resolve_cli_audio_specs(args.audio, cfg)
        if args.geocode:
            cfg.trips.geocode = True
        if args.auto_climax:
            cfg.video.auto_climax = True
        if args.climax_sec is not None:
            cfg.video.climax_duration_sec = args.climax_sec
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = _run_scan_and_trips(photos_dir, cfg)
        manifest_target = (
            Path(cfg.project.manifest) if cfg.project.manifest else out_dir / "manifest.json"
        )
        manifest_path = manifest.save(manifest_target)
        _print_scan_summary(manifest, manifest_path)

        mosaic_path: Path | None = None
        if target_path is not None:
            tile_paths = [p.path for p in manifest.photos if p.selected_for_mosaic]
            mosaic_path = out_dir / "mosaic.jpg"
            m_res = build_mosaic(target_path, tile_paths, mosaic_path, config=cfg.mosaic)
            logger.info(
                "Mosaic saved: %s (%dx%dpx)",
                m_res.output_path,
                m_res.width_px,
                m_res.height_px,
            )

        v_res = render_keepsake_video(
            manifest,
            out_dir / "keepsake.mp4",
            config=cfg.video,
            audio_specs=audio_specs or None,
            mosaic_image_path=mosaic_path,
            mosaic_cols=cfg.mosaic.cols,
        )
        manifest.save(manifest_path)
        logger.info("Video saved: %s (%.1fs)", v_res.output_path, v_res.duration_sec)
        if v_res.pacing_plan is not None:
            _print_pacing_summary(v_res.pacing_plan)
        return 0

    return 1


def _run_scan_and_trips(photos_dir: Path, cfg: ProjectConfig) -> Manifest:
    manifest = scan_photos(photos_dir, cfg.ingest)
    cluster_trips(manifest, cfg.trips)
    select_video_photos(manifest, cfg.ingest.max_video_photos)
    return manifest


def _override_mosaic_config(base: MosaicConfig, args: argparse.Namespace) -> MosaicConfig:
    if args.paper is not None:
        base.paper_size = args.paper
    if args.dpi is not None:
        base.dpi = args.dpi
    if args.cols is not None:
        base.cols = args.cols
    if args.color_correction is not None:
        base.color_correction = args.color_correction
    if args.overlay_alpha is not None:
        base.overlay_alpha = args.overlay_alpha
    return base


def _resolve_cli_audio_specs(
    cli_audio: list[str] | None,
    cfg: ProjectConfig,
) -> list[AudioTrackSpec]:
    if cli_audio is not None:
        return expand_audio_inputs(list(cli_audio))
    return cfg.project.resolved_audio_specs()


def _resolve_mosaic_finale(
    args: argparse.Namespace,
    cfg: ProjectConfig,
) -> Path | None:
    if getattr(args, "no_mosaic_finale", False):
        return None
    if getattr(args, "mosaic_finale", None) is not None:
        return args.mosaic_finale
    if args.config and cfg.video.mosaic_finale_sec > 0:
        candidate = cfg.project.resolved_mosaic_path()
        if candidate.exists():
            return candidate
    return None


def _override_video_config(base: VideoConfig, args: argparse.Namespace) -> VideoConfig:
    if args.resolution is not None:
        base.resolution = args.resolution
    if args.duration is not None:
        base.target_duration_sec = args.duration
    if args.no_title_cards:
        base.include_title_cards = False
    if getattr(args, "auto_climax", False):
        base.auto_climax = True
    if getattr(args, "climax_sec", None) is not None:
        base.climax_duration_sec = args.climax_sec
    if getattr(args, "pacing", None) is not None:
        base.pacing_mode = args.pacing
    return base


def _print_pacing_summary(plan: PacingPlan, manifest_path: Path | None = None) -> None:
    suffix = f" Saved to {manifest_path}" if manifest_path else ""
    logger.info(
        "Pacing plan: %d photos (~%.1fs) -> %d burst, %d standard, %d hero "
        "(clip range %.2fs–%.2fs, avg %.2fs).%s",
        len(plan.clips),
        plan.total_photo_span_sec,
        plan.burst_count,
        plan.standard_count,
        plan.hero_count,
        plan.min_clip_sec,
        plan.max_clip_sec,
        plan.avg_clip_sec,
        suffix,
    )


def _print_scan_summary(manifest: Manifest, out_path: Path) -> None:
    mosaic_count = sum(1 for p in manifest.photos if p.selected_for_mosaic)
    video_count = sum(1 for p in manifest.photos if p.selected_for_video)
    logger.info(
        "Scanned %d photos -> %d curated for mosaic, %d selected for video, "
        "%d chapter(s). Manifest: %s",
        len(manifest.photos),
        mosaic_count,
        video_count,
        len(manifest.trips),
        out_path,
    )
