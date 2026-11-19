"""Wallpaper-style Ken Burns video builder with title cards, batched xfade, and audio."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from keepsake._audio import (
    AudioTimeline,
    AudioTrackSpec,
    ResolvedAudioSegment,
    analyze_audio_dynamics,
    estimate_segments_duration,
    expand_audio_inputs,
    mix_audio_segments,
    probe_media_duration,
    resolve_audio_segments,
)
from keepsake._config import VideoConfig
from keepsake._image_io import open_image
from keepsake._logging import get_logger
from keepsake._models import Manifest, PhotoRecord, TripChapter
from keepsake._mosaic import render_mosaic_puzzle_finale, render_mosaic_zoom_reveal
from keepsake._pacing import PacingPlan, PlannedClip, plan_video_pacing

logger = get_logger(__name__)


@dataclass(slots=True)
class ClipSpec:
    """Internal representation of a rendered video segment before joining."""

    path: Path
    duration_sec: float
    transition_after_sec: float = 0.0


@dataclass(slots=True)
class VideoResult:
    """Summary of a rendered keepsake video."""

    output_path: Path
    duration_sec: float
    width: int
    height: int
    fps: int
    photo_count: int
    chapter_count: int
    has_audio: bool
    audio_segments: list[ResolvedAudioSegment] = field(default_factory=list)
    pacing_plan: PacingPlan | None = None


def render_keepsake_video(
    manifest: Manifest,
    output_path: str | Path,
    config: VideoConfig | None = None,
    *,
    audio_paths: list[str | Path] | None = None,
    audio_specs: list[AudioTrackSpec] | None = None,
    mosaic_image_path: str | Path | None = None,
    mosaic_cols: int | None = None,
) -> VideoResult:
    """Render a dynamic music-synced keepsake video from curated photos in `manifest`."""
    _require_ffmpeg()
    cfg = config or VideoConfig()
    width, height = cfg.resolved_dimensions()

    photos = [p for p in manifest.photos if p.selected_for_video]
    if not photos:
        raise ValueError("No photos marked with selected_for_video=True in manifest.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve audio tracks (with optional climax slicing) to determine target duration
    specs = (
        list(audio_specs)
        if audio_specs is not None
        else expand_audio_inputs(list(audio_paths or []))
    )
    missing_specs = [s for s in specs if not s.path.exists()]
    for missing in missing_specs:
        logger.warning("Audio file not found, skipping: %s", missing.path)
    valid_specs = [s for s in specs if s.path.exists()]
    segments = (
        resolve_audio_segments(
            valid_specs,
            auto_climax=cfg.auto_climax,
            climax_duration_sec=cfg.climax_duration_sec,
            crossfade_sec=cfg.audio_crossfade_sec,
        )
        if valid_specs
        else []
    )
    target_total = cfg.target_duration_sec
    if segments and cfg.seconds_per_photo is None:
        audio_dur = estimate_segments_duration(segments, crossfade_sec=cfg.audio_crossfade_sec)
        if audio_dur > 0:
            target_total = audio_dur

    has_quote = bool(cfg.closing_quote and cfg.closing_quote.strip()) and cfg.closing_quote_sec > 0
    has_finale = mosaic_image_path is not None and cfg.mosaic_finale_sec > 0
    effective_card_mode = cfg.title_card_mode if cfg.include_title_cards else "none"

    with tempfile.TemporaryDirectory(prefix="keepsake_vid_") as tmp_str:
        tmp_dir = Path(tmp_str)
        raw_mixed_wav: Path | None = None
        audio_timeline: AudioTimeline | None = None

        if segments:
            raw_mixed_wav = tmp_dir / "mixed_raw.wav"
            mix_audio_segments(
                segments,
                raw_mixed_wav,
                crossfade_sec=cfg.audio_crossfade_sec,
                fade_in_sec=cfg.audio_fade_in_sec,
                fade_out_sec=0.0,
                normalize=cfg.audio_normalize,
                target_duration_sec=None,
            )
            audio_timeline = analyze_audio_dynamics(raw_mixed_wav, segments=segments)
            if cfg.seconds_per_photo is None:
                actual_audio_dur = probe_media_duration(raw_mixed_wav)
                if actual_audio_dur > 0:
                    target_total = actual_audio_dur

        # Estimate title-card, closing quote, and finale overhead
        num_cards = _estimate_title_card_count(
            manifest.photos if cfg.dynamic_fill_from_curated else photos,
            manifest.trips,
            mode=effective_card_mode,
        )
        card_overhead = num_cards * max(0.0, cfg.title_card_sec - cfg.transition_sec)
        quote_overhead = max(0.0, cfg.closing_quote_sec - cfg.transition_sec) if has_quote else 0.0
        finale_overhead = (
            max(0.0, cfg.mosaic_finale_sec - cfg.transition_sec) if has_finale else 0.0
        )
        target_photo_span = max(
            2.0, target_total - card_overhead - quote_overhead - finale_overhead
        )

        plan = _resolve_pacing_plan(
            manifest,
            photos,
            target_photo_span,
            cfg,
            timeline=audio_timeline,
        )

        sections = _group_planned_for_cards(
            plan.clips,
            manifest.trips,
            mode=effective_card_mode,
            year_prefix=cfg.year_card_prefix,
        )
        if segments and cfg.seconds_per_photo is None:
            _align_plan_to_target_duration(
                sections,
                target_total=target_total,
                cfg=cfg,
                has_quote=has_quote,
                has_finale=has_finale,
            )
        clips: list[ClipSpec] = []
        photo_idx = 0

        for card_title, card_sub, section_items in sections:
            if card_title:
                card_img = tmp_dir / f"card_{len(clips):04d}.png"
                render_title_card_image(
                    card_title,
                    card_sub,
                    card_img,
                    width=width,
                    height=height,
                )
                card_clip = tmp_dir / f"clip_{len(clips):04d}.mp4"
                actual_card_dur = _render_still_clip(
                    card_img,
                    card_clip,
                    width=width,
                    height=height,
                    duration_sec=cfg.title_card_sec,
                    fps=cfg.fps,
                    codec=cfg.codec,
                    crf=cfg.crf,
                    preset=cfg.preset,
                )
                clips.append(
                    ClipSpec(
                        card_clip,
                        actual_card_dur,
                        min(cfg.transition_sec, actual_card_dur * 0.4),
                    )
                )

            for item in section_items:
                composed = tmp_dir / f"frame_{len(clips):04d}.png"
                compose_photo_frame(
                    item.photo.path,
                    composed,
                    width=width,
                    height=height,
                    portrait_mode=cfg.portrait_mode,
                )
                clip_path = tmp_dir / f"clip_{len(clips):04d}.mp4"
                if cfg.camera_motion == "none":
                    actual_dur = _render_still_clip(
                        composed,
                        clip_path,
                        width=width,
                        height=height,
                        duration_sec=item.duration_sec,
                        fps=cfg.fps,
                        codec=cfg.codec,
                        crf=cfg.crf,
                        preset=cfg.preset,
                    )
                elif item.duration_sec < 0.55:
                    if cfg.motion_sync and item.energy >= 0.45:
                        actual_dur = _render_beat_punch_clip(
                            composed,
                            clip_path,
                            width=width,
                            height=height,
                            duration_sec=item.duration_sec,
                            fps=cfg.fps,
                            punch_in=(photo_idx % 2) == 0,
                            strength=0.025 + 0.025 * item.energy,
                            codec=cfg.codec,
                            crf=cfg.crf,
                            preset=cfg.preset,
                        )
                    else:
                        actual_dur = _render_still_clip(
                            composed,
                            clip_path,
                            width=width,
                            height=height,
                            duration_sec=item.duration_sec,
                            fps=cfg.fps,
                            codec=cfg.codec,
                            crf=cfg.crf,
                            preset=cfg.preset,
                        )
                else:
                    eff_zoom = (
                        cfg.zoom_strength * (0.55 + 0.65 * item.energy)
                        if cfg.motion_sync
                        else cfg.zoom_strength
                    )
                    allow_pan = cfg.camera_motion == "subtle_pan" and not item.photo.is_portrait
                    motion_style = (photo_idx % 5) if allow_pan else (photo_idx % 2)
                    actual_dur = _render_ken_burns_clip(
                        composed,
                        clip_path,
                        width=width,
                        height=height,
                        duration_sec=item.duration_sec,
                        fps=cfg.fps,
                        motion_style=motion_style,
                        zoom_strength=eff_zoom,
                        codec=cfg.codec,
                        crf=cfg.crf,
                        preset=cfg.preset,
                    )
                clips.append(ClipSpec(clip_path, actual_dur, item.transition_after_sec))
                photo_idx += 1

        if has_quote and cfg.closing_quote:
            quote_img = tmp_dir / f"quote_{len(clips):04d}.png"
            render_quote_card_image(
                cfg.closing_quote.strip(),
                quote_img,
                width=width,
                height=height,
            )
            quote_clip = tmp_dir / f"clip_{len(clips):04d}_quote.mp4"
            actual_quote_dur = _render_ken_burns_clip(
                quote_img,
                quote_clip,
                width=width,
                height=height,
                duration_sec=cfg.closing_quote_sec,
                fps=cfg.fps,
                motion_style=0,
                zoom_strength=0.04,
                codec=cfg.codec,
                crf=cfg.crf,
                preset=cfg.preset,
            )
            quote_trans = min(cfg.transition_sec, actual_quote_dur * 0.35) if has_finale else 0.0
            clips.append(ClipSpec(quote_clip, actual_quote_dur, quote_trans))

        if has_finale and mosaic_image_path is not None:
            finale_clip = tmp_dir / f"clip_{len(clips):04d}_finale.mp4"
            if cfg.mosaic_finale_style == "puzzle":
                render_mosaic_puzzle_finale(
                    mosaic_image_path,
                    finale_clip,
                    width=width,
                    height=height,
                    duration_sec=cfg.mosaic_finale_sec,
                    fps=cfg.fps,
                    cols=mosaic_cols,
                    pattern=cfg.mosaic_puzzle_pattern,
                    focal_x=cfg.mosaic_focal_x,
                    focal_y=cfg.mosaic_focal_y,
                    codec=cfg.codec,
                    crf=cfg.crf,
                    preset=cfg.preset,
                )
            else:
                render_mosaic_zoom_reveal(
                    mosaic_image_path,
                    finale_clip,
                    width=width,
                    height=height,
                    duration_sec=cfg.mosaic_finale_sec,
                    fps=cfg.fps,
                    focal_x=cfg.mosaic_focal_x,
                    focal_y=cfg.mosaic_focal_y,
                    codec=cfg.codec,
                    crf=cfg.crf,
                    preset=cfg.preset,
                )
            clips.append(ClipSpec(finale_clip, cfg.mosaic_finale_sec, 0.0))

        merged_video = tmp_dir / "merged_silent.mp4"
        final_dur = _join_clips_hybrid(
            clips,
            merged_video,
            batch_size=max(2, cfg.batch_size),
            codec=cfg.codec,
            crf=cfg.crf,
            preset=cfg.preset,
            work_dir=tmp_dir,
        )

        if raw_mixed_wav is not None:
            _mux_premixed_audio(
                merged_video,
                raw_mixed_wav,
                out_path,
                video_duration_sec=final_dur,
                fade_out_sec=cfg.audio_fade_out_sec,
                codec=cfg.codec,
            )
        else:
            _finalize_silent_mp4(merged_video, out_path, codec=cfg.codec)

    return VideoResult(
        output_path=out_path,
        duration_sec=round(final_dur, 2),
        width=width,
        height=height,
        fps=cfg.fps,
        photo_count=len(plan.clips),
        chapter_count=len(sections),
        has_audio=bool(segments),
        audio_segments=segments,
        pacing_plan=plan,
    )


def compose_photo_frame(
    photo_path: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    portrait_mode: str = "blurred_bg",
) -> Path:
    """Compose a photo onto a `width x height` frame without distortion."""
    out = Path(output_path)
    with open_image(photo_path, mode="RGB") as img:
        if portrait_mode == "crop":
            canvas = ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)
        elif portrait_mode == "pad":
            canvas = Image.new("RGB", (width, height), (14, 16, 20))
            fg = ImageOps.contain(img, (width, height), method=Image.Resampling.LANCZOS)
            canvas.paste(fg, ((width - fg.width) // 2, (height - fg.height) // 2))
        else:
            # Blurred atmospheric background + crisp centered foreground
            bg = ImageOps.fit(img, (width, height), method=Image.Resampling.BILINEAR)
            blur_radius = max(18, width // 55)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            bg = ImageEnhance.Brightness(bg).enhance(0.45)

            margin_ratio = 0.94 if (img.height > img.width) else 1.0
            max_w = round(width * margin_ratio)
            max_h = round(height * margin_ratio)
            fg = ImageOps.contain(img, (max_w, max_h), method=Image.Resampling.LANCZOS)
            bg.paste(fg, ((width - fg.width) // 2, (height - fg.height) // 2))
            canvas = bg

    canvas.save(out, format="PNG")
    return out


def render_title_card_image(
    title: str,
    subtitle: str,
    output_path: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Render a minimal typography title card as a PNG image."""
    img = Image.new("RGB", (width, height), (16, 18, 24))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(size=max(28, round(height * 0.052)))
    sub_font = _load_font(size=max(18, round(height * 0.028)))

    cx, cy = width // 2, height // 2
    draw.text(
        (cx, cy - round(height * 0.025)),
        title,
        fill=(242, 242, 245),
        font=title_font,
        anchor="mm",
    )
    if subtitle:
        draw.text(
            (cx, cy + round(height * 0.045)),
            subtitle,
            fill=(170, 178, 194),
            font=sub_font,
            anchor="mm",
        )

    line_half = round(width * 0.06)
    line_y = cy + round(height * 0.01)
    draw.line(
        [(cx - line_half, line_y), (cx + line_half, line_y)],
        fill=(120, 130, 150),
        width=max(2, height // 540),
    )

    out = Path(output_path)
    img.save(out, format="PNG")
    return out


def render_quote_card_image(
    quote: str,
    output_path: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Render an atmospheric closing quote card with centered serif typography."""
    img = Image.new("RGB", (width, height), (12, 14, 19))
    draw = ImageDraw.Draw(img)

    font_size = max(28, round(height * 0.056))
    font = _load_quote_font(size=font_size)

    # Word-wrap smoothly to stay within 78% of frame width
    max_text_w = round(width * 0.78)
    words = quote.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if current and (bbox[2] - bbox[0]) > max_text_w:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    line_gap = round(font_size * 1.45)
    total_h = line_gap * max(1, len(lines))
    start_y = (height - total_h) // 2 + line_gap // 2
    cx = width // 2

    for idx, line in enumerate(lines):
        draw.text(
            (cx, start_y + idx * line_gap),
            line,
            fill=(244, 244, 248),
            font=font,
            anchor="mm",
        )

    accent_y = start_y + len(lines) * line_gap + round(height * 0.025)
    accent_half = round(width * 0.045)
    draw.line(
        [(cx - accent_half, accent_y), (cx + accent_half, accent_y)],
        fill=(140, 148, 168),
        width=max(2, height // 540),
    )

    out = Path(output_path)
    img.save(out, format="PNG")
    return out


def total_audio_duration(audio_paths: list[Path], *, crossfade_sec: float = 2.0) -> float:
    """Estimate combined duration of multiple audio tracks after crossfades."""
    durations = [probe_media_duration(p) for p in audio_paths]
    if not durations:
        return 0.0
    total = sum(durations)
    overlap = max(0, len(durations) - 1) * min(crossfade_sec, min(durations) * 0.4)
    return max(0.0, total - overlap)


def _resolve_pacing_plan(
    manifest: Manifest,
    photos: list[PhotoRecord],
    target_photo_span: float,
    cfg: VideoConfig,
    *,
    timeline: AudioTimeline | None,
) -> PacingPlan:
    if cfg.seconds_per_photo is None and all(p.duration_sec is not None for p in photos):
        clips: list[PlannedClip] = []
        cursor = 0.0
        for idx, rec in enumerate(photos):
            dur = float(rec.duration_sec or 1.5)
            trans = (
                float(rec.transition_sec)
                if rec.transition_sec is not None
                else min(cfg.transition_sec, dur * 0.35)
            )
            clips.append(
                PlannedClip(
                    photo=rec,
                    duration_sec=dur,
                    transition_after_sec=trans
                    if idx < len(photos) - 1
                    else min(cfg.transition_sec, dur * 0.35),
                    role=rec.video_role or "standard",
                    reason=rec.pacing_reason or "manifest override",
                    start_time_sec=round(cursor, 3),
                    energy=float(rec.audio_energy) if rec.audio_energy is not None else 0.5,
                )
            )
            cursor += max(0.1, dur - (trans if idx < len(photos) - 1 else 0.0))
        durs = [c.duration_sec for c in clips]
        return PacingPlan(
            clips=clips,
            total_photo_span_sec=round(cursor, 2),
            burst_count=sum(1 for c in clips if c.role == "burst"),
            standard_count=sum(1 for c in clips if c.role == "standard"),
            hero_count=sum(1 for c in clips if c.role == "hero"),
            min_clip_sec=round(min(durs), 3),
            max_clip_sec=round(max(durs), 3),
            avg_clip_sec=round(sum(durs) / len(durs), 3),
        )

    return plan_video_pacing(
        manifest,
        target_photo_span,
        cfg,
        timeline=timeline,
        persist_to_manifest=True,
    )


def _render_still_clip(
    image_path: Path,
    out_path: Path,
    *,
    width: int,
    height: int,
    duration_sec: float,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
) -> float:
    total_frames = max(2, round(duration_sec * fps))
    actual_dur = total_frames / float(fps)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        f"scale={width}:{height},fps={fps},format=yuv420p",
        "-frames:v",
        str(total_frames),
        "-r",
        str(fps),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-video_track_timescale",
        "90000",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return actual_dur


def _render_beat_punch_clip(
    image_path: Path,
    out_path: Path,
    *,
    width: int,
    height: int,
    duration_sec: float,
    fps: int,
    punch_in: bool,
    strength: float,
    codec: str,
    crf: int,
    preset: str,
) -> float:
    """Fast 1x-scale micro zoom-punch synced to high-energy beat cuts."""
    total_frames = max(2, round(duration_sec * fps))
    actual_dur = total_frames / float(fps)
    peak_zoom = 1.0 + max(0.02, min(0.10, strength))
    step = (peak_zoom - 1.0) / max(1, total_frames - 1)
    if punch_in:
        z_expr = f"min({peak_zoom:.5f},1+{step:.7f}*on)"
    else:
        z_expr = f"max(1.0,{peak_zoom:.5f}-{step:.7f}*on)"

    vf = (
        f"scale={width}:{height}:flags=bilinear,"
        f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(image_path),
        "-vf",
        vf,
        "-frames:v",
        str(total_frames),
        "-r",
        str(fps),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-video_track_timescale",
        "90000",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return actual_dur


def _render_ken_burns_clip(
    image_path: Path,
    out_path: Path,
    *,
    width: int,
    height: int,
    duration_sec: float,
    fps: int,
    motion_style: int,
    zoom_strength: float,
    codec: str,
    crf: int,
    preset: str,
) -> float:
    total_frames = max(2, round(duration_sec * fps))
    actual_dur = total_frames / float(fps)
    max_zoom = 1.0 + max(0.02, min(0.20, zoom_strength))
    denom = max(1, total_frames - 1)
    step = (max_zoom - 1.0) / denom

    style = motion_style % 5
    zoom_in = style in (0, 2, 3)
    if zoom_in:
        z_expr = f"min({max_zoom:.5f},1+{step:.7f}*on)"
    else:
        z_expr = f"max(1.0,{max_zoom:.5f}-{step:.7f}*on)"

    if style == 2:
        # Subtle drift left to right
        x_expr = f"(iw-iw/zoom)*(0.42+0.16*on/{denom})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == 3:
        # Subtle drift right to left
        x_expr = f"(iw-iw/zoom)*(0.58-0.16*on/{denom})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif style == 4:
        # Subtle upward drift
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(ih-ih/zoom)*(0.58-0.16*on/{denom})"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    # Upscale 2x before zoompan to eliminate integer rounding jitter on slow zooms
    up_w, up_h = width * 2, height * 2
    vf = (
        f"scale={up_w}:{up_h}:flags=lanczos,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(image_path),
        "-vf",
        vf,
        "-frames:v",
        str(total_frames),
        "-r",
        str(fps),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-video_track_timescale",
        "90000",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return actual_dur


def _join_clips_hybrid(
    clips: list[ClipSpec],
    out_path: Path,
    *,
    batch_size: int,
    codec: str,
    crf: int,
    preset: str,
    work_dir: Path,
) -> float:
    if len(clips) == 1:
        shutil.copy2(clips[0].path, out_path)
        return clips[0].duration_sec

    collapsed = _collapse_hard_cuts(clips, work_dir=work_dir)
    if len(collapsed) == 1:
        shutil.copy2(collapsed[0].path, out_path)
        return collapsed[0].duration_sec

    level = 0
    current = collapsed
    while len(current) > 1:
        next_level: list[ClipSpec] = []
        for idx in range(0, len(current), batch_size):
            chunk = current[idx : idx + batch_size]
            if len(chunk) == 1:
                next_level.append(chunk[0])
                continue
            is_final = len(current) <= batch_size
            target = out_path if is_final else work_dir / f"merge_l{level}_{idx}.mp4"
            dur = _xfade_batch(
                chunk,
                target,
                codec=codec,
                crf=crf,
                preset=preset,
            )
            next_level.append(ClipSpec(target, dur, chunk[-1].transition_after_sec))
        current = next_level
        level += 1
    return current[0].duration_sec


def _collapse_hard_cuts(clips: list[ClipSpec], *, work_dir: Path) -> list[ClipSpec]:
    """Concatenate consecutive hard-cut clips (`transition_after_sec <= 0.05`) without re-encoding."""
    collapsed: list[ClipSpec] = []
    run: list[ClipSpec] = []

    def _flush_run(group: list[ClipSpec], run_idx: int) -> None:
        if len(group) == 1:
            collapsed.append(group[0])
            return
        list_file = work_dir / f"concat_{run_idx}.txt"
        lines = [
            f"file '{str(c.path.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            for c in group
        ]
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        merged = work_dir / f"concat_run_{run_idx}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(merged),
        ]
        _run_ffmpeg(cmd)
        collapsed.append(
            ClipSpec(
                merged,
                sum(c.duration_sec for c in group),
                group[-1].transition_after_sec,
            )
        )

    run_counter = 0
    for idx, clip in enumerate(clips):
        run.append(clip)
        if clip.transition_after_sec > 0.05 or idx == len(clips) - 1:
            _flush_run(run, run_counter)
            run_counter += 1
            run = []
    return collapsed


def _xfade_batch(
    clips: list[ClipSpec],
    out_path: Path,
    *,
    codec: str,
    crf: int,
    preset: str,
) -> float:
    cmd = ["ffmpeg", "-y"]
    for clip in clips:
        cmd.extend(["-i", str(clip.path)])

    filter_parts: list[str] = []
    running_dur = clips[0].duration_sec
    prev_label = "[0:v]"

    for i in range(1, len(clips)):
        requested = clips[i - 1].transition_after_sec
        safe_trans = max(
            0.04,
            min(requested, clips[i - 1].duration_sec * 0.4, clips[i].duration_sec * 0.4),
        )
        offset = max(0.0, running_dur - safe_trans)
        out_label = "[vout]" if i == len(clips) - 1 else f"[vx{i}]"
        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition=fade:duration={safe_trans:.3f}:"
            f"offset={offset:.3f}{out_label}"
        )
        running_dur = offset + clips[i].duration_sec
        prev_label = out_label

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-video_track_timescale",
            "90000",
            str(out_path),
        ]
    )
    _run_ffmpeg(cmd)
    return running_dur


def _predict_sequence_duration(
    sections: list[tuple[str, str, list[PlannedClip]]],
    cfg: VideoConfig,
    *,
    has_quote: bool,
    has_finale: bool,
) -> float:
    """Predict exact joined video duration matching `_join_clips_hybrid` frame-quantized math."""
    fps = max(1, cfg.fps)

    def _q(dur: float) -> float:
        return max(2, round(dur * fps)) / float(fps)

    raw: list[tuple[float, float]] = []
    for card_title, _, items in sections:
        if card_title:
            cd = _q(cfg.title_card_sec)
            raw.append((cd, min(cfg.transition_sec, cd * 0.4)))
        for item in items:
            raw.append((_q(item.duration_sec), item.transition_after_sec))
    if has_quote:
        qd = _q(cfg.closing_quote_sec)
        raw.append((qd, min(cfg.transition_sec, qd * 0.35) if has_finale else 0.0))
    if has_finale:
        raw.append((_q(cfg.mosaic_finale_sec), 0.0))
    if not raw:
        return 0.0

    # Mirror `_collapse_hard_cuts`
    collapsed: list[tuple[float, float]] = []
    run_dur = 0.0
    for idx, (dur, trans) in enumerate(raw):
        run_dur += dur
        if trans > 0.05 or idx == len(raw) - 1:
            collapsed.append((run_dur, trans))
            run_dur = 0.0

    total = collapsed[0][0]
    for i in range(1, len(collapsed)):
        prev_dur, prev_trans = collapsed[i - 1]
        curr_dur, _ = collapsed[i]
        safe_trans = max(0.04, min(prev_trans, prev_dur * 0.4, curr_dur * 0.4))
        total = max(0.0, total - safe_trans) + curr_dur
    return total


def _align_plan_to_target_duration(
    sections: list[tuple[str, str, list[PlannedClip]]],
    *,
    target_total: float,
    cfg: VideoConfig,
    has_quote: bool,
    has_finale: bool,
) -> None:
    """Adjust photo clip durations before rendering so total video matches exact audio length."""
    all_items = [item for _, _, items in sections for item in items]
    if not all_items:
        return
    fps = max(1, cfg.fps)
    min_d = max(0.15, cfg.min_clip_sec)
    max_d = max(min_d + 0.2, cfg.max_clip_sec * 1.25)

    for tier_roles in (("hero",), ("standard",), ("hero", "standard"), ("burst",)):
        predicted = _predict_sequence_duration(
            sections, cfg, has_quote=has_quote, has_finale=has_finale
        )
        drift = target_total - predicted
        if abs(drift) <= 0.06:
            break
        candidates = [it for it in all_items if it.role in tier_roles]
        for _ in range(5):
            if abs(drift) <= 0.06 or not candidates:
                break
            delta = drift / len(candidates)
            next_cands: list[PlannedClip] = []
            for item in candidates:
                floor = max(0.8, min_d) if item.role == "hero" else min_d
                ceil = max_d if item.role != "burst" else max(0.65, min(max_d, 1.2))
                frames = max(
                    max(2, round(floor * fps)),
                    round(max(floor, min(ceil, item.duration_sec + delta)) * fps),
                )
                new_dur = round(frames / float(fps), 4)
                if new_dur != item.duration_sec:
                    item.duration_sec = new_dur
                    item.photo.duration_sec = round(new_dur, 3)
                    if item.transition_after_sec > 0.05:
                        new_tr = round(min(item.transition_after_sec, new_dur * 0.38), 3)
                        item.transition_after_sec = 0.0 if new_tr <= 0.05 else new_tr
                        item.photo.transition_sec = item.transition_after_sec
                    if floor < new_dur < ceil:
                        next_cands.append(item)
            predicted = _predict_sequence_duration(
                sections, cfg, has_quote=has_quote, has_finale=has_finale
            )
            drift = target_total - predicted
            candidates = next_cands


def _mp4_compat_args(codec: str) -> list[str]:
    """Return Apple/iCloud/QuickTime MP4 compatibility flags (+faststart, BT.709, codec tag)."""
    args: list[str] = []
    if codec == "libx264":
        args.extend(
            [
                "-bsf:v",
                "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                "-tag:v",
                "avc1",
            ]
        )
    elif codec == "libx265":
        args.extend(["-tag:v", "hvc1"])
    args.extend(["-movflags", "+faststart"])
    return args


def _finalize_silent_mp4(video_path: Path, out_path: Path, *, codec: str = "libx264") -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "copy",
        *_mp4_compat_args(codec),
        str(out_path),
    ]
    _run_ffmpeg(cmd)


def _mux_premixed_audio(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    video_duration_sec: float,
    fade_out_sec: float,
    codec: str = "libx264",
) -> None:
    actual_audio_dur = probe_media_duration(audio_path)
    fade_end = (
        min(video_duration_sec, actual_audio_dur) if actual_audio_dur > 0 else video_duration_sec
    )
    safe_out = min(max(0.0, fade_out_sec), fade_end * 0.4)
    af = f"apad,atrim=0:{video_duration_sec:.3f}"
    if safe_out > 0.02:
        fade_start = max(0.0, fade_end - safe_out)
        af += f",afade=t=out:st={fade_start:.3f}:d={safe_out:.3f}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-af",
        af,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-shortest",
        *_mp4_compat_args(codec),
        str(out_path),
    ]
    _run_ffmpeg(cmd)


def _estimate_title_card_count(
    photos: list[PhotoRecord],
    trips: list[TripChapter],
    *,
    mode: str,
) -> int:
    active = [p for p in photos if p.selected_for_mosaic or p.selected_for_video]
    if not active or mode == "none":
        return 0
    if mode == "year":
        years = {p.parsed_timestamp.year for p in active if p.parsed_timestamp is not None}
        return len(years)
    return len(_group_photos_by_chapter(active, trips))


def _group_planned_for_cards(
    clips: list[PlannedClip],
    trips: list[TripChapter],
    *,
    mode: str,
    year_prefix: str = "Year ",
) -> list[tuple[str, str, list[PlannedClip]]]:
    if mode == "none":
        return [("", "", clips)]
    if mode == "year":
        sections: list[tuple[str, str, list[PlannedClip]]] = []
        current_year: int | None = None
        current_group: list[PlannedClip] = []
        for clip in clips:
            yr = clip.photo.parsed_timestamp.year if clip.photo.parsed_timestamp else None
            if current_group and yr != current_year:
                label = f"{year_prefix}{current_year}" if current_year is not None else ""
                sections.append((label, "", current_group))
                current_group = []
            current_year = yr
            current_group.append(clip)
        if current_group:
            label = f"{year_prefix}{current_year}" if current_year is not None else ""
            sections.append((label, "", current_group))
        return sections
    return _group_planned_by_chapter(clips, trips)


def _group_planned_by_chapter(
    clips: list[PlannedClip], trips: list[TripChapter]
) -> list[tuple[str, str, list[PlannedClip]]]:
    if not trips:
        return [("Memories", "", clips)]

    trip_map = {t.index: t for t in trips}
    grouped: dict[int | None, list[PlannedClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.photo.trip_index, []).append(clip)

    chapters: list[tuple[str, str, list[PlannedClip]]] = []
    for trip_idx, group in grouped.items():
        trip = trip_map.get(trip_idx) if trip_idx is not None else None
        if trip is not None:
            parts = trip.title.split(" — ", maxsplit=1)
            title = parts[0]
            subtitle = parts[1] if len(parts) > 1 else ""
        else:
            title = "Memories"
            subtitle = ""
        chapters.append((title, subtitle, group))
    return chapters


def _group_photos_by_chapter(
    photos: list[PhotoRecord], trips: list[TripChapter]
) -> list[tuple[str, str, list[PhotoRecord]]]:
    if not trips:
        return [("Memories", "", photos)]

    trip_map = {t.index: t for t in trips}
    grouped: dict[int | None, list[PhotoRecord]] = {}
    for photo in photos:
        grouped.setdefault(photo.trip_index, []).append(photo)

    chapters: list[tuple[str, str, list[PhotoRecord]]] = []
    for trip_idx, group in grouped.items():
        trip = trip_map.get(trip_idx) if trip_idx is not None else None
        if trip is not None:
            parts = trip.title.split(" — ", maxsplit=1)
            title = parts[0]
            subtitle = parts[1] if len(parts) > 1 else ""
        else:
            title = "Memories"
            subtitle = ""
        chapters.append((title, subtitle, group))
    return chapters


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _load_quote_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Italic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH.")


def _run_ffmpeg(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({res.returncode}): {res.stderr[-700:]}")
