"""Audio climax detection, slicing, loudness-normalized mixing, and beat/energy analysis."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHORTHAND_RE = re.compile(
    r"^(?P<path>.+?)@(?P<spec>(?:auto|full|\d+(?::\d+)*(?:\.\d+)?-\d+(?::\d+)*(?:\.\d+)?))$",
    re.IGNORECASE,
)

AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus"})


@dataclass(slots=True)
class AudioTrackSpec:
    """User-specified audio file with optional explicit start/end trim timestamps."""

    path: Path
    start_sec: float | None = None
    end_sec: float | None = None
    force_full: bool = False


@dataclass(slots=True)
class ResolvedAudioSegment:
    """Concrete trimmed audio window and its placement in the mixed soundtrack."""

    path: Path
    source_duration_sec: float
    start_sec: float
    end_sec: float
    mix_start_sec: float
    mix_end_sec: float
    auto_detected: bool = False
    peak_sec: float | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class AudioTimeline:
    """Energy envelope and beat/onset timestamps extracted from a mixed soundtrack."""

    duration_sec: float
    step_sec: float
    energy: list[float]
    beat_times: list[float]
    segments: list[ResolvedAudioSegment]


def split_audio_shorthand(raw: str) -> tuple[str, str | None]:
    """Split optional `@start-end` / `@auto` / `@full` suffix from an audio path string."""
    match = _SHORTHAND_RE.match(raw.strip())
    if not match:
        return raw.strip(), None
    return match.group("path"), match.group("spec")


def parse_timestamp_sec(value: str | float | int | None) -> float | None:
    """Parse seconds from float/int or `MM:SS` / `HH:MM:SS` string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid timestamp format: '{value}'")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return max(0.0, total)


def parse_audio_spec(raw: str | Path) -> AudioTrackSpec:
    """Parse `path` or `path@start-end` / `path@auto` / `path@full` into AudioTrackSpec."""
    if isinstance(raw, Path):
        return AudioTrackSpec(path=raw)
    path_str, spec = split_audio_shorthand(raw)
    path = Path(path_str)
    if spec is None:
        return AudioTrackSpec(path=path)
    lower = spec.lower()
    if lower == "auto":
        return AudioTrackSpec(path=path)
    if lower == "full":
        return AudioTrackSpec(path=path, force_full=True)
    start_s, end_s = spec.split("-", maxsplit=1)
    return AudioTrackSpec(
        path=path,
        start_sec=parse_timestamp_sec(start_s),
        end_sec=parse_timestamp_sec(end_s),
    )


def discover_audio_files(directory: str | Path) -> list[Path]:
    """Return sorted audio files in `directory` matching AUDIO_EXTENSIONS."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Expected audio directory, got file: {dir_path}")
    files = [
        p
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def expand_audio_inputs(items: list[str | Path]) -> list[AudioTrackSpec]:
    """Expand a mix of audio file paths, `@start-end` specs, and audio directories."""
    specs: list[AudioTrackSpec] = []
    for item in items:
        if isinstance(item, Path) and item.is_dir():
            specs.extend(AudioTrackSpec(path=p) for p in discover_audio_files(item))
            continue
        if isinstance(item, str) and Path(item).is_dir():
            specs.extend(AudioTrackSpec(path=p) for p in discover_audio_files(item))
            continue
        specs.append(parse_audio_spec(item))
    return specs


def format_timestamp(seconds: float) -> str:
    """Format seconds as `MM:SS`."""
    whole = max(0, round(seconds))
    mins, secs = divmod(whole, 60)
    return f"{mins:02d}:{secs:02d}"


def probe_media_duration(path: str | Path) -> float:
    """Return media duration in seconds using ffprobe."""
    _require_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    return float(data["format"]["duration"])


def decode_audio_mono_pcm(path: str | Path, *, sample_rate: int = 16000) -> np.ndarray:
    """Decode an audio file to normalized mono float32 PCM samples via ffmpeg."""
    _require_ffmpeg()
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, check=True)
    if not res.stdout:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def detect_climax_window(
    path: str | Path,
    target_duration_sec: float = 40.0,
    *,
    hop_sec: float = 0.25,
) -> tuple[float, float, float]:
    """Locate the highest-energy climax window `[start_sec, end_sec]` and its peak time."""
    sample_rate = 16000
    samples = decode_audio_mono_pcm(path, sample_rate=sample_rate)
    if samples.size == 0:
        dur = probe_media_duration(path)
        end = min(dur, target_duration_sec) if target_duration_sec > 0 else dur
        return 0.0, end, end * 0.5

    total_dur = samples.size / float(sample_rate)
    if target_duration_sec <= 0 or total_dur <= target_duration_sec + 1.0:
        peak_sec = _find_peak_rms_sec(samples, sample_rate=sample_rate, hop_sec=hop_sec)
        return 0.0, round(total_dur, 3), round(peak_sec, 3)

    hop_samples = max(1, round(sample_rate * hop_sec))
    usable = (samples.size // hop_samples) * hop_samples
    frames = samples[:usable].reshape(-1, hop_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    rms_max = float(np.percentile(rms, 98)) or float(np.max(rms)) or 1.0
    rms_norm = np.clip(rms / rms_max, 0.0, 1.5)
    flux = np.maximum(0.0, np.diff(rms_norm, prepend=rms_norm[0]))
    flux_max = float(np.max(flux)) or 1.0
    score = 0.75 * rms_norm + 0.25 * (flux / flux_max)

    win_hops = max(1, min(len(score), round(target_duration_sec / hop_sec)))
    cumsum = np.concatenate(([0.0], np.cumsum(score)))
    window_scores = cumsum[win_hops:] - cumsum[:-win_hops]

    # Slightly prefer non-intro windows if intro and chorus score similarly
    if len(window_scores) > 10:
        intro_cutoff = max(1, len(window_scores) // 10)
        window_scores[:intro_cutoff] *= 0.92

    best_idx = int(np.argmax(window_scores))
    start_sec = best_idx * hop_sec
    end_sec = min(total_dur, start_sec + target_duration_sec)
    local_rms = rms[best_idx : best_idx + win_hops]
    peak_offset = int(np.argmax(local_rms)) * hop_sec if local_rms.size else 0.0
    peak_sec = min(end_sec, start_sec + peak_offset)
    return round(start_sec, 3), round(end_sec, 3), round(peak_sec, 3)


def resolve_audio_segments(
    specs: list[AudioTrackSpec],
    *,
    auto_climax: bool = False,
    climax_duration_sec: float = 40.0,
    crossfade_sec: float = 2.0,
) -> list[ResolvedAudioSegment]:
    """Resolve track specs into concrete `[start, end]` windows and mix timeline offsets."""
    resolved: list[ResolvedAudioSegment] = []
    cursor = 0.0

    for spec in specs:
        src_dur = probe_media_duration(spec.path)
        auto_detected = False
        peak_sec: float | None = None

        if spec.force_full:
            start_sec, end_sec = 0.0, src_dur
        elif spec.start_sec is not None or spec.end_sec is not None:
            start_sec = max(0.0, min(spec.start_sec or 0.0, max(0.0, src_dur - 0.5)))
            if spec.end_sec is not None:
                end_sec = min(src_dur, max(start_sec + 0.5, spec.end_sec))
            elif auto_climax and climax_duration_sec > 0:
                end_sec = min(src_dur, start_sec + climax_duration_sec)
            else:
                end_sec = src_dur
        elif auto_climax and climax_duration_sec > 0:
            start_sec, end_sec, peak_sec = detect_climax_window(
                spec.path, target_duration_sec=climax_duration_sec
            )
            auto_detected = True
        else:
            start_sec, end_sec = 0.0, src_dur

        seg_dur = max(0.5, end_sec - start_sec)
        if not resolved:
            mix_start = 0.0
        else:
            prev_dur = resolved[-1].duration_sec
            safe_xfade = _safe_crossfade(crossfade_sec, prev_dur, seg_dur)
            mix_start = max(0.0, cursor - safe_xfade)

        mix_end = mix_start + seg_dur
        cursor = mix_end
        resolved.append(
            ResolvedAudioSegment(
                path=spec.path,
                source_duration_sec=round(src_dur, 3),
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                mix_start_sec=round(mix_start, 3),
                mix_end_sec=round(mix_end, 3),
                auto_detected=auto_detected,
                peak_sec=peak_sec,
            )
        )

    return resolved


def estimate_segments_duration(
    segments: list[ResolvedAudioSegment], *, crossfade_sec: float = 2.0
) -> float:
    """Return total duration after sequential crossfades across `segments`."""
    if not segments:
        return 0.0
    cursor = segments[0].duration_sec
    for prev, curr in zip(segments, segments[1:], strict=False):
        safe_xfade = _safe_crossfade(crossfade_sec, prev.duration_sec, curr.duration_sec)
        cursor = max(0.0, cursor - safe_xfade) + curr.duration_sec
    return round(cursor, 3)


def mix_audio_segments(
    segments: list[ResolvedAudioSegment],
    output_path: str | Path,
    *,
    crossfade_sec: float = 2.0,
    fade_in_sec: float = 1.0,
    fade_out_sec: float = 3.0,
    normalize: bool = True,
    target_duration_sec: float | None = None,
) -> float:
    """Slice, loudness-normalize, crossfade, and write the combined audio track."""
    if not segments:
        raise ValueError("segments must contain at least one audio track.")
    _require_ffmpeg()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    for seg in segments:
        cmd.extend(["-i", str(seg.path)])

    filter_parts: list[str] = []
    norm_labels: list[str] = []
    for idx, seg in enumerate(segments):
        label = f"[s{idx}]"
        chain = [
            f"atrim=start={seg.start_sec:.3f}:end={seg.end_sec:.3f}",
            "asetpts=PTS-STARTPTS",
        ]
        if normalize:
            chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        chain.append("aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo")
        filter_parts.append(f"[{idx}:a]{','.join(chain)}{label}")
        norm_labels.append(label)

    if len(norm_labels) == 1:
        mixed_label = norm_labels[0]
    else:
        prev_label = norm_labels[0]
        for idx in range(1, len(norm_labels)):
            safe_xfade = _safe_crossfade(
                crossfade_sec,
                segments[idx - 1].duration_sec,
                segments[idx].duration_sec,
            )
            out_label = f"[ax{idx}]"
            filter_parts.append(
                f"{prev_label}{norm_labels[idx]}acrossfade=d={safe_xfade:.3f}:c1=tri:c2=tri{out_label}"
            )
            prev_label = out_label
        mixed_label = prev_label

    estimated_dur = estimate_segments_duration(segments, crossfade_sec=crossfade_sec)
    final_dur = target_duration_sec if target_duration_sec is not None else estimated_dur

    tail_filters: list[str] = []
    if target_duration_sec is not None:
        tail_filters.append(f"apad,atrim=0:{final_dur:.3f}")

    safe_in = min(max(0.0, fade_in_sec), final_dur * 0.25)
    safe_out = min(max(0.0, fade_out_sec), final_dur * 0.4)
    if safe_in > 0.02:
        tail_filters.append(f"afade=t=in:st=0:d={safe_in:.3f}")
    if safe_out > 0.02:
        fade_start = max(0.0, final_dur - safe_out)
        tail_filters.append(f"afade=t=out:st={fade_start:.3f}:d={safe_out:.3f}")

    if tail_filters:
        filter_parts.append(f"{mixed_label}{','.join(tail_filters)}[aout]")
    else:
        filter_parts.append(f"{mixed_label}anull[aout]")

    cmd.extend(["-filter_complex", ";".join(filter_parts), "-map", "[aout]"])
    if out_path.suffix.lower() == ".wav":
        cmd.extend(["-c:a", "pcm_s16le"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "256k"])
    cmd.append(str(out_path))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed ({res.returncode}): {res.stderr[-700:]}")
    return final_dur


def analyze_audio_dynamics(
    audio_path: str | Path,
    segments: list[ResolvedAudioSegment] | None = None,
    *,
    step_sec: float = 0.1,
    min_beat_gap_sec: float = 0.20,
) -> AudioTimeline:
    """Compute normalized RMS energy curve and transient beat timestamps from audio."""
    sample_rate = 16000
    samples = decode_audio_mono_pcm(audio_path, sample_rate=sample_rate)
    if samples.size == 0:
        dur = probe_media_duration(audio_path)
        return AudioTimeline(
            duration_sec=dur,
            step_sec=step_sec,
            energy=[],
            beat_times=[],
            segments=segments or [],
        )

    duration_sec = samples.size / float(sample_rate)
    hop_samples = max(1, round(sample_rate * step_sec))
    usable = (samples.size // hop_samples) * hop_samples
    frames = samples[:usable].reshape(-1, hop_samples)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    ref_level = float(np.percentile(rms, 98)) or float(np.max(rms)) or 1.0
    energy_norm = np.clip(rms / ref_level, 0.0, 1.0)

    # Spectral flux on 20 ms frames for beat / transient onset detection
    beat_times = _detect_beats_spectral_flux(
        samples,
        sample_rate=sample_rate,
        min_gap_sec=min_beat_gap_sec,
    )

    return AudioTimeline(
        duration_sec=round(duration_sec, 3),
        step_sec=step_sec,
        energy=[round(float(v), 4) for v in energy_norm],
        beat_times=beat_times,
        segments=segments or [],
    )


def _detect_beats_spectral_flux(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_sec: float = 0.02,
    min_gap_sec: float = 0.20,
) -> list[float]:
    hop = max(64, round(sample_rate * frame_sec))
    win_len = hop * 2
    if samples.size < win_len:
        return []

    n_frames = 1 + (samples.size - win_len) // hop
    indices = np.arange(win_len)[None, :] + (np.arange(n_frames) * hop)[:, None]
    windowed = samples[indices] * np.hanning(win_len)[None, :]
    mags = np.abs(np.fft.rfft(windowed, axis=1))
    diff = np.maximum(0.0, np.diff(mags, axis=0, prepend=mags[:1]))
    flux = np.sum(diff, axis=1)

    if flux.size < 3:
        return []

    # Adaptive local threshold so quiet builds and loud drops both yield crisp beats
    kernel = max(5, round(0.4 / frame_sec))
    pad = kernel // 2
    padded = np.pad(flux, pad, mode="edge")
    local_mean = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")[: flux.size]
    global_thresh = float(np.mean(flux) + 0.65 * np.std(flux))
    threshold = np.maximum(local_mean * 1.35, global_thresh * 0.5)

    peaks: list[int] = []
    min_gap_frames = max(1, round(min_gap_sec / frame_sec))
    for i in range(1, flux.size - 1):
        if flux[i] >= threshold[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap_frames:
                peaks.append(i)
            elif flux[i] > flux[peaks[-1]]:
                peaks[-1] = i

    return [round((idx * hop) / float(sample_rate), 3) for idx in peaks]


def _find_peak_rms_sec(samples: np.ndarray, *, sample_rate: int, hop_sec: float) -> float:
    hop = max(1, round(sample_rate * hop_sec))
    usable = (samples.size // hop) * hop
    if usable == 0:
        return 0.0
    frames = samples[:usable].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return float(np.argmax(rms)) * hop_sec


def _safe_crossfade(crossfade_sec: float, prev_dur: float, curr_dur: float) -> float:
    return max(0.0, min(crossfade_sec, prev_dur * 0.4, curr_dur * 0.4))


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH.")
