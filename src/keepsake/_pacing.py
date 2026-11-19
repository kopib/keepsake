"""Context-aware variable photo pacing driven by audio dynamics and photo metadata."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from keepsake._audio import AudioTimeline
from keepsake._config import VideoConfig
from keepsake._curate import select_video_photos
from keepsake._metadata import hamming_distance
from keepsake._models import Manifest, PhotoRecord


@dataclass(slots=True)
class PlannedClip:
    """Pacing assignment for a single photo in the video timeline."""

    photo: PhotoRecord
    duration_sec: float
    transition_after_sec: float
    role: str
    reason: str
    start_time_sec: float
    energy: float = 0.5


@dataclass(slots=True)
class PacingPlan:
    """Full variable-pacing timeline across selected photos."""

    clips: list[PlannedClip]
    total_photo_span_sec: float
    burst_count: int
    standard_count: int
    hero_count: int
    min_clip_sec: float
    max_clip_sec: float
    avg_clip_sec: float


@dataclass(slots=True)
class _PhotoContext:
    photo: PhotoRecord
    importance: float
    is_anchor: bool
    is_isolated: bool
    is_dense: bool
    sharpness_pct: float
    is_burst_member: bool
    reason_tags: list[str]


def plan_video_pacing(
    manifest: Manifest,
    target_photo_span_sec: float,
    config: VideoConfig,
    *,
    timeline: AudioTimeline | None = None,
    persist_to_manifest: bool = True,
) -> PacingPlan:
    """Assign non-random per-photo durations and transitions from photo + audio context."""
    target_span = max(0.5, float(target_photo_span_sec))
    fps = max(1, config.fps)

    if config.seconds_per_photo is not None or config.pacing_mode == "uniform":
        return _build_uniform_plan(
            manifest,
            target_span,
            config,
            persist_to_manifest=persist_to_manifest,
        )

    photos = _choose_dynamic_photo_pool(
        manifest,
        target_span,
        config,
        timeline=timeline,
    )
    if not photos:
        raise ValueError("No photos available for video pacing.")

    contexts = _score_photo_contexts(
        manifest,
        photos,
        title_card_mode=config.title_card_mode if config.include_title_cards else "none",
    )
    min_d = max(0.15, config.min_clip_sec)
    max_d = max(min_d + 0.2, config.max_clip_sec)
    beats = timeline.beat_times if (timeline and config.beat_sync) else []
    e_lo, e_hi = _energy_bounds(timeline)

    raw_durations: list[float] = []
    raw_transitions: list[float] = []
    roles: list[str] = []
    reasons: list[str] = []
    energies: list[float] = []

    t = 0.0
    n = len(contexts)
    for idx, ctx in enumerate(contexts):
        remaining_clips = n - idx
        remaining_time = max(min_d * remaining_clips, target_span - t)
        avg_remaining = remaining_time / remaining_clips

        progress = min(1.0, t / target_span) if target_span > 0 else 0.0
        raw_energy = _sample_energy(timeline, t, progress)
        rel_energy = _normalize_energy(raw_energy, e_lo, e_hi)
        beat_density = _local_beat_density(beats, t)

        role, dur, trans, mode_tag = _assign_slot(
            ctx,
            energy=rel_energy,
            beat_density=beat_density,
            avg_remaining=avg_remaining,
            min_d=min_d,
            max_d=max_d,
            base_transition=config.transition_sec,
            t=t,
            beats=beats,
            is_last=(idx == n - 1),
        )

        dur = _quantize_to_frame(dur, fps, minimum=min_d)
        if trans <= 0.05:
            trans = 0.0
        else:
            trans = round(min(trans, dur * 0.38), 3)

        raw_durations.append(dur)
        raw_transitions.append(trans)
        roles.append(role)
        energies.append(rel_energy)
        tags = [*ctx.reason_tags, mode_tag]
        reasons.append("; ".join(dict.fromkeys(tags)))

        t += max(0.1, dur - (trans if idx < n - 1 else 0.0))

    # Enforce neighbor-safe transitions so ffmpeg xfade never shrinks overlaps unexpectedly
    _clamp_transitions_to_neighbors(raw_durations, raw_transitions)

    # Absorb any residual drift cleanly into hero/standard clips (keeping burst cuts crisp)
    _rebalance_durations(
        raw_durations,
        raw_transitions,
        roles,
        target_span=target_span,
        min_d=min_d,
        max_d=max_d * 1.15,
        fps=fps,
    )

    clips: list[PlannedClip] = []
    cursor = 0.0
    for idx, ctx in enumerate(contexts):
        dur = raw_durations[idx]
        trans = raw_transitions[idx] if idx < n - 1 else min(config.transition_sec, dur * 0.35)
        clips.append(
            PlannedClip(
                photo=ctx.photo,
                duration_sec=dur,
                transition_after_sec=trans,
                role=roles[idx],
                reason=reasons[idx],
                start_time_sec=round(cursor, 3),
                energy=round(energies[idx], 3),
            )
        )
        cursor += max(0.1, dur - (trans if idx < n - 1 else 0.0))

    if persist_to_manifest:
        _write_plan_to_manifest(manifest, clips)

    return _summarize_plan(clips, total_span=cursor)


def _build_uniform_plan(
    manifest: Manifest,
    target_span: float,
    config: VideoConfig,
    *,
    persist_to_manifest: bool,
) -> PacingPlan:
    photos = [p for p in manifest.photos if p.selected_for_video]
    if not photos:
        raise ValueError("No photos marked with selected_for_video=True in manifest.")

    n = len(photos)
    if config.seconds_per_photo is not None:
        dur = config.seconds_per_photo
    else:
        overlap = max(0, n - 1) * config.transition_sec
        dur = max(2.5, (target_span + overlap) / n)

    clips: list[PlannedClip] = []
    cursor = 0.0
    for idx, rec in enumerate(photos):
        trans = (
            min(config.transition_sec, dur * 0.4)
            if idx < n - 1
            else min(config.transition_sec, dur * 0.4)
        )
        clips.append(
            PlannedClip(
                photo=rec,
                duration_sec=round(dur, 3),
                transition_after_sec=round(trans, 3),
                role="standard",
                reason="uniform pacing",
                start_time_sec=round(cursor, 3),
            )
        )
        cursor += max(0.1, dur - (trans if idx < n - 1 else 0.0))

    if persist_to_manifest:
        _write_plan_to_manifest(manifest, clips)
    return _summarize_plan(clips, total_span=cursor)


def _choose_dynamic_photo_pool(
    manifest: Manifest,
    target_span: float,
    config: VideoConfig,
    *,
    timeline: AudioTimeline | None,
) -> list[PhotoRecord]:
    selected = [p for p in manifest.photos if p.selected_for_video]
    curated = [p for p in manifest.photos if p.selected_for_mosaic]
    if not selected:
        selected = list(curated)

    if not config.dynamic_fill_from_curated or len(curated) <= len(selected):
        return _inject_burst_sequences(manifest, selected, target_span, config)

    # Estimate natural average clip duration from target_avg_clip_sec & audio energy
    mean_energy = (
        sum(timeline.energy) / len(timeline.energy) if (timeline and timeline.energy) else 0.55
    )
    desired_avg = max(
        config.min_clip_sec + 0.15,
        min(
            config.max_clip_sec * 0.8,
            config.target_avg_clip_sec * (1.12 - 0.25 * mean_energy),
        ),
    )
    needed_count = round(target_span / desired_avg)
    if needed_count <= len(selected):
        return _inject_burst_sequences(manifest, selected, target_span, config)

    target_count = min(len(curated), max(len(selected), needed_count))
    expanded = select_video_photos(manifest, target_count)
    return _inject_burst_sequences(manifest, expanded, target_span, config)


def _inject_burst_sequences(
    manifest: Manifest,
    base_photos: list[PhotoRecord],
    target_span: float,
    config: VideoConfig,
) -> list[PhotoRecord]:
    """Pull up to 2-3 companion frames from burst groups for rapid flipbook cuts."""
    if target_span < 12.0 or config.min_clip_sec > 0.5:
        return base_photos

    by_burst: dict[int, list[PhotoRecord]] = {}
    for rec in manifest.photos:
        if rec.burst_group_id is not None:
            by_burst.setdefault(rec.burst_group_id, []).append(rec)

    selected_paths = {p.path for p in base_photos}
    extra_budget = max(0, min(48, round(target_span / 5.0)))
    added: list[PhotoRecord] = []

    for rec in base_photos:
        if rec.burst_group_id is None or extra_budget <= 0:
            continue
        siblings = [
            s
            for s in by_burst.get(rec.burst_group_id, ())
            if s.path not in selected_paths and s.sharpness >= rec.sharpness * 0.65
        ]
        # Keep up to 2 companion frames per burst so flashes stay punchy (~0.6-0.9s per group)
        for sib in siblings[:2]:
            if extra_budget <= 0:
                break
            selected_paths.add(sib.path)
            added.append(sib)
            extra_budget -= 1

    if not added:
        return base_photos

    combined = [p for p in manifest.photos if p.path in selected_paths]
    combined.sort(key=lambda r: (r.timestamp or "9999", r.path))
    return combined


def _score_photo_contexts(
    manifest: Manifest,
    photos: list[PhotoRecord],
    *,
    title_card_mode: str = "year",
) -> list[_PhotoContext]:
    sharp_values = sorted(p.sharpness for p in photos)
    n = len(photos)

    # Identify anchor bounds matching the active grouping (year vs chapter)
    anchor_paths: set[str] = set()
    if n > 0:
        anchor_paths.add(photos[0].path)
        anchor_paths.add(photos[-1].path)

    if title_card_mode == "year":
        by_year: dict[int | None, list[str]] = {}
        for p in photos:
            yr = p.parsed_timestamp.year if p.parsed_timestamp else None
            by_year.setdefault(yr, []).append(p.path)
        for paths in by_year.values():
            anchor_paths.add(paths[0])
            anchor_paths.add(paths[-1])
    elif title_card_mode == "chapter":
        by_trip: dict[int, list[str]] = {}
        for p in photos:
            if p.trip_index is not None:
                by_trip.setdefault(p.trip_index, []).append(p.path)
        for paths in by_trip.values():
            anchor_paths.add(paths[0])
            anchor_paths.add(paths[-1])

    burst_counts: dict[int, int] = {}
    for p in photos:
        if p.burst_group_id is not None:
            burst_counts[p.burst_group_id] = burst_counts.get(p.burst_group_id, 0) + 1

    contexts: list[_PhotoContext] = []
    for idx, rec in enumerate(photos):
        tags: list[str] = []
        sharp_rank = bisect.bisect_right(sharp_values, rec.sharpness) / max(1, len(sharp_values))

        is_anchor = rec.path in anchor_paths
        dt_nearest_min = _nearest_neighbor_minutes(photos, idx)
        is_isolated = dt_nearest_min is not None and dt_nearest_min >= 180.0
        is_dense = dt_nearest_min is not None and dt_nearest_min <= 12.0

        is_burst = rec.burst_group_id is not None and burst_counts.get(rec.burst_group_id, 0) >= 2

        novelty = 0.5
        if idx > 0 and rec.dhash and photos[idx - 1].dhash:
            novelty = hamming_distance(rec.dhash, photos[idx - 1].dhash) / 64.0

        score = 0.35 * sharp_rank + 0.12 * (0.0 if rec.is_portrait else 1.0) + 0.13 * novelty
        if is_anchor:
            score += 0.25
            tags.append("year anchor" if title_card_mode == "year" else "chapter anchor")
        if is_isolated:
            score += 0.18
            tags.append("unique moment")
        if is_dense:
            score -= 0.16
            tags.append("dense sequence")
        if sharp_rank >= 0.75:
            tags.append("sharp shot")
        if not rec.is_portrait:
            tags.append("16:9 friendly")
        if is_burst:
            score -= 0.22
            tags.append("kinetic burst")

        importance = max(0.0, min(1.0, score))
        contexts.append(
            _PhotoContext(
                photo=rec,
                importance=importance,
                is_anchor=is_anchor,
                is_isolated=is_isolated,
                is_dense=is_dense,
                sharpness_pct=sharp_rank,
                is_burst_member=is_burst,
                reason_tags=tags or ["balanced moment"],
            )
        )
    return contexts


def _assign_slot(
    ctx: _PhotoContext,
    *,
    energy: float,
    beat_density: float,
    avg_remaining: float,
    min_d: float,
    max_d: float,
    base_transition: float,
    t: float,
    beats: list[float],
    is_last: bool,
) -> tuple[str, float, float, str]:
    # Burst micro-sequences always play as rapid kinetic cuts
    if ctx.is_burst_member:
        target = min(0.36, max(min_d, min_d + 0.12 * (1.0 - energy)))
        dur = _snap_to_beat(t, target, beats, min_d=min_d, max_d=0.42)
        return "burst", dur, 0.0, "kinetic burst cut"

    # High-energy chorus peaks or rapid beat clusters trigger crisp 0.20s-0.45s cuts
    if (energy >= 0.58 or beat_density >= 2.2) and not ctx.is_anchor and ctx.importance < 0.72:
        target = min(0.48, max(min_d, 0.44 - 0.22 * max(0.0, energy - 0.55) / 0.45))
        dur = _snap_to_beat(t, target, beats, min_d=min_d, max_d=0.52)
        return "burst", dur, 0.0, f"high audio energy ({energy:.2f})"

    # Quiet vocal builds or year anchors earn expressive Ken Burns hero holds
    if (
        (ctx.is_anchor and energy < 0.75)
        or (ctx.importance >= 0.70 and energy < 0.52)
        or energy < 0.26
    ):
        base = max(
            1.45,
            min(max_d, avg_remaining * 1.45 + 0.65 * ctx.importance - 0.35 * energy),
        )
        dur = _snap_to_beat(t, base, beats, min_d=1.35, max_d=max_d)
        trans = (
            min(base_transition, dur * 0.35) if not is_last else min(base_transition, dur * 0.35)
        )
        return "hero", dur, trans, f"hero hold (energy {energy:.2f})"

    # Standard music-paced shot
    base = max(0.55, min(1.55, avg_remaining * (1.15 - 0.50 * energy) + 0.25 * ctx.importance))
    dur = _snap_to_beat(t, base, beats, min_d=0.50, max_d=1.75)
    if energy >= 0.44 or dur < 0.85:
        trans = 0.0
    else:
        trans = min(base_transition * 0.55, dur * 0.28)
    return "standard", dur, trans, f"beat-paced (energy {energy:.2f})"


def _clamp_transitions_to_neighbors(
    durations: list[float],
    transitions: list[float],
) -> None:
    """Ensure every transition fits both adjacent clips (hard-cutting into/out of rapid flashes)."""
    n = len(durations)
    for i in range(n - 1):
        if transitions[i] <= 0.05 or durations[i] < 0.55 or durations[i + 1] < 0.55:
            transitions[i] = 0.0
        else:
            clamped = round(min(transitions[i], durations[i] * 0.38, durations[i + 1] * 0.38), 3)
            transitions[i] = 0.0 if clamped <= 0.05 else clamped


def _rebalance_durations(
    durations: list[float],
    transitions: list[float],
    roles: list[str],
    *,
    target_span: float,
    min_d: float,
    max_d: float,
    fps: int,
) -> None:
    n = len(durations)
    _clamp_transitions_to_neighbors(durations, transitions)
    current = sum(durations) - sum(transitions[:-1]) if n > 1 else sum(durations)
    diff = target_span - current
    if abs(diff) <= 0.08:
        return

    # Adjust hero clips first, spilling over into standard (and finally all) if saturated
    tiers: tuple[tuple[str, ...], ...] = (
        ("hero",),
        ("standard",),
        ("hero", "standard"),
        ("burst",),
    )
    for tier_roles in tiers:
        if abs(diff) <= 0.05:
            break
        adjustable = [i for i, r in enumerate(roles) if r in tier_roles]
        for _ in range(4):
            if abs(diff) <= 0.05 or not adjustable:
                break
            delta = diff / len(adjustable)
            next_adj: list[int] = []
            for idx in adjustable:
                floor = min_d if roles[idx] != "hero" else max(1.0, min_d)
                ceil = max_d if roles[idx] != "burst" else max(0.65, min(max_d, 1.2))
                proposed = _quantize_to_frame(
                    max(floor, min(ceil, durations[idx] + delta)), fps, minimum=floor
                )
                if proposed != durations[idx]:
                    durations[idx] = proposed
                    if floor < proposed < ceil:
                        next_adj.append(idx)
            _clamp_transitions_to_neighbors(durations, transitions)
            current = sum(durations) - sum(transitions[:-1]) if n > 1 else sum(durations)
            diff = target_span - current
            adjustable = next_adj


def _sample_energy(
    timeline: AudioTimeline | None,
    t: float,
    progress: float,
) -> float:
    if timeline is None or not timeline.energy:
        # Smooth dramatic arc when rendering without audio analysis
        if progress < 0.20:
            return 0.30 + progress * 1.5
        if progress < 0.78:
            return 0.60 + 0.30 * abs((progress * 4.0) % 2.0 - 1.0)
        return max(0.25, 0.80 - (progress - 0.78) * 2.5)

    step = max(0.01, timeline.step_sec)
    center = int(round(t / step))
    radius = max(1, round(0.6 / step))
    lo = max(0, center - radius)
    hi = min(len(timeline.energy), center + radius + 1)
    if lo >= hi:
        return timeline.energy[-1]
    return float(sum(timeline.energy[lo:hi]) / (hi - lo))


def _snap_to_beat(
    t: float,
    target_dur: float,
    beats: list[float],
    *,
    min_d: float,
    max_d: float,
) -> float:
    if not beats:
        return max(min_d, min(max_d, target_dur))
    desired_end = t + target_dur
    lo_idx = bisect.bisect_left(beats, t + min_d)
    hi_idx = bisect.bisect_right(beats, t + max_d)
    if lo_idx >= hi_idx:
        return max(min_d, min(max_d, target_dur))

    best_beat = min(beats[lo_idx:hi_idx], key=lambda b: abs(b - desired_end))
    # Only snap if a beat sits reasonably close to the intended rhythm
    if abs(best_beat - desired_end) <= max(0.18, target_dur * 0.35):
        return max(min_d, min(max_d, best_beat - t))
    return max(min_d, min(max_d, target_dur))


def _nearest_neighbor_minutes(photos: list[PhotoRecord], idx: int) -> float | None:
    curr = photos[idx].parsed_timestamp
    if curr is None:
        return None
    gaps: list[float] = []
    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(photos):
            dt = photos[neighbor_idx].parsed_timestamp
            if dt is not None:
                gaps.append(abs((curr - dt).total_seconds()) / 60.0)
    return min(gaps) if gaps else None


def _quantize_to_frame(duration_sec: float, fps: int, *, minimum: float) -> float:
    frames = max(max(2, round(minimum * fps)), round(duration_sec * fps))
    return round(frames / float(fps), 4)


def _energy_bounds(timeline: AudioTimeline | None) -> tuple[float, float]:
    if timeline is None or len(timeline.energy) < 5:
        return 0.0, 1.0
    vals = sorted(timeline.energy)
    lo = vals[max(0, int(len(vals) * 0.10))]
    hi = vals[min(len(vals) - 1, int(len(vals) * 0.92))]
    if hi - lo < 0.10:
        return 0.0, 1.0
    return lo, hi


def _normalize_energy(raw_energy: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (raw_energy - lo) / max(0.10, hi - lo)))


def _local_beat_density(beats: list[float], t: float, window_sec: float = 3.0) -> float:
    if not beats:
        return 0.0
    half = window_sec * 0.5
    lo = bisect.bisect_left(beats, t - half)
    hi = bisect.bisect_right(beats, t + half)
    return (hi - lo) / window_sec


def _write_plan_to_manifest(manifest: Manifest, clips: list[PlannedClip]) -> None:
    by_path = {c.photo.path: c for c in clips}
    for rec in manifest.photos:
        planned = by_path.get(rec.path)
        if planned is None:
            rec.selected_for_video = False
            rec.video_role = None
            rec.duration_sec = None
            rec.transition_sec = None
            rec.pacing_reason = None
            rec.audio_energy = None
        else:
            rec.selected_for_video = True
            rec.video_role = planned.role
            rec.duration_sec = round(planned.duration_sec, 3)
            rec.transition_sec = round(planned.transition_after_sec, 3)
            rec.pacing_reason = planned.reason
            rec.audio_energy = round(planned.energy, 3)


def _summarize_plan(clips: list[PlannedClip], *, total_span: float) -> PacingPlan:
    durs = [c.duration_sec for c in clips]
    return PacingPlan(
        clips=clips,
        total_photo_span_sec=round(total_span, 2),
        burst_count=sum(1 for c in clips if c.role == "burst"),
        standard_count=sum(1 for c in clips if c.role == "standard"),
        hero_count=sum(1 for c in clips if c.role == "hero"),
        min_clip_sec=round(min(durs), 3) if durs else 0.0,
        max_clip_sec=round(max(durs), 3) if durs else 0.0,
        avg_clip_sec=round(sum(durs) / len(durs), 3) if durs else 0.0,
    )
