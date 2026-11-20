"""Tests for keepsake metadata, curation, trip clustering, mosaic, video, and CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageDraw

from keepsake import (
    IngestConfig,
    Manifest,
    MosaicConfig,
    ProjectConfig,
    TripConfig,
    VideoConfig,
    build_mosaic,
    cluster_trips,
    configure_logging,
    get_logger,
    plan_video_pacing,
    render_keepsake_video,
    render_mosaic_puzzle_finale,
    render_mosaic_zoom_reveal,
    scan_photos,
    select_video_photos,
)
from keepsake._audio import (
    analyze_audio_dynamics,
    detect_climax_window,
    parse_audio_spec,
    parse_timestamp_sec,
    probe_media_duration,
)
from keepsake._metadata import _read_exif_dict, compute_sharpness, extract_photo_record
from keepsake.cli import main as cli_main


def _make_textured_image(
    path: Path,
    size: tuple[int, int] = (640, 480),
    *,
    base_rgb: tuple[int, int, int] = (120, 160, 200),
    sharp: bool = True,
    exif_bytes: bytes | None = None,
) -> Path:
    rng = np.random.default_rng(abs(hash((path.name, base_rgb, sharp))) % (2**32))
    arr = np.full((size[1], size[0], 3), base_rgb, dtype=np.int16)
    if sharp:
        noise = rng.integers(-45, 46, size=(size[1], size[0], 3), dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if sharp:
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(255, 255, 255), width=4)
        draw.line([0, 0, size[0], size[1]], fill=(30, 30, 30), width=5)
    save_kwargs = {"exif": exif_bytes} if exif_bytes else {}
    img.save(path, format="JPEG", quality=92, **save_kwargs)
    return path


def _build_exif(
    *,
    dt_str: str,
    make: str = "Apple",
    model: str = "iPhone 15",
    lat_dms: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
    lat_ref: str = "N",
    lon_dms: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
    lon_ref: str = "E",
) -> bytes:
    exif = Image.Exif()
    exif[271] = make
    exif[272] = model
    exif[306] = dt_str
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[36867] = dt_str
    if lat_dms is not None and lon_dms is not None:
        from PIL.TiffImagePlugin import IFDRational

        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[1] = lat_ref
        gps_ifd[2] = tuple(IFDRational(n, d) for n, d in lat_dms)
        gps_ifd[3] = lon_ref
        gps_ifd[4] = tuple(IFDRational(n, d) for n, d in lon_dms)
    return exif.tobytes()


class _FakeGeocoder:
    def reverse(self, lat: float, lon: float) -> str | None:
        if lat > 45:
            return "Munich, Germany"
        return "Lisbon, Portugal"


def test_exif_gps_and_camera_offset(tmp_path: Path) -> None:
    exif = _build_exif(
        dt_str="2025:06:10 14:30:00",
        make="Canon",
        model="EOS R6",
        lat_dms=((38, 1), (43, 1), (0, 1)),
        lat_ref="N",
        lon_dms=((9, 1), (8, 1), (0, 1)),
        lon_ref="W",
    )
    img_path = _make_textured_image(tmp_path / "lisbon.jpg", exif_bytes=exif)
    rec = extract_photo_record(img_path, camera_time_offsets_hours={"canon": -2.0})

    assert rec.camera_make == "Canon"
    assert rec.camera_model == "EOS R6"
    assert rec.timestamp == "2025-06-10T12:30:00"
    assert rec.latitude == pytest.approx(38.716667, abs=1e-4)
    assert rec.longitude == pytest.approx(-9.133333, abs=1e-4)
    assert rec.sharpness > 50.0


def test_curation_burst_dedup_and_sharpness(tmp_path: Path) -> None:
    # Two near-identical frames (solid color, slightly noisy vs smooth -> same dhash)
    img_blur = _make_textured_image(tmp_path / "01_blur.jpg", base_rgb=(100, 120, 140), sharp=False)
    # Create a sharp image with identical coarse layout for burst test
    arr = np.full((240, 320, 3), (100, 120, 140), dtype=np.uint8)
    arr[::2, ::2] = (115, 135, 155)
    img_sharp = tmp_path / "02_sharp.jpg"
    Image.fromarray(arr).save(img_sharp)

    assert compute_sharpness(Image.open(img_sharp)) > compute_sharpness(Image.open(img_blur))

    cfg = IngestConfig(min_sharpness=0.0, dedup_hash_distance=16)
    manifest = scan_photos(tmp_path, cfg)
    selected = [p for p in manifest.photos if p.selected_for_mosaic]
    assert len(selected) == 1
    assert selected[0].path.endswith("02_sharp.jpg")


def test_trip_clustering_and_balanced_selection(tmp_path: Path) -> None:
    # Trip 1: Lisbon (3 photos)
    for i in range(3):
        exif = _build_exif(
            dt_str=f"2025:05:0{i + 1} 12:00:00",
            lat_dms=((38, 1), (43, 1), (0, 1)),
            lat_ref="N",
            lon_dms=((9, 1), (8, 1), (0, 1)),
            lon_ref="W",
        )
        _make_textured_image(
            tmp_path / f"t1_{i}.jpg",
            base_rgb=(200, 120 + i * 25, 80),
            exif_bytes=exif,
        )

    # Trip 2: Munich 10 days later (3 photos)
    for i in range(3):
        exif = _build_exif(
            dt_str=f"2025:05:1{i + 2} 15:00:00",
            lat_dms=((48, 1), (8, 1), (0, 1)),
            lat_ref="N",
            lon_dms=((11, 1), (34, 1), (0, 1)),
            lon_ref="E",
        )
        _make_textured_image(
            tmp_path / f"t2_{i}.jpg",
            base_rgb=(60, 130 + i * 25, 210),
            exif_bytes=exif,
        )

    manifest = scan_photos(tmp_path, IngestConfig(dedup_hash_distance=-1, min_sharpness=0.0))
    trips = cluster_trips(manifest, TripConfig(), geocoder=_FakeGeocoder())

    assert len(trips) == 2
    assert "Lisbon, Portugal" in trips[0].title
    assert "Munich, Germany" in trips[1].title

    chosen = select_video_photos(manifest, max_photos=4)
    assert len(chosen) == 4
    trip_ids = [p.trip_index for p in chosen]
    assert trip_ids.count(1) == 2
    assert trip_ids.count(2) == 2


def test_mosaic_a4_portrait_and_zoom_end_to_end(tmp_path: Path) -> None:
    # Target portrait (vertical 3:4)
    target_path = tmp_path / "target.jpg"
    img = Image.new("RGB", (600, 800), (40, 80, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([150, 180, 450, 520], fill=(230, 185, 160))
    img.save(target_path)

    # Diverse tile colors
    tile_paths: list[Path] = []
    colors = [
        (40, 80, 160),
        (230, 185, 160),
        (200, 90, 80),
        (80, 160, 90),
        (240, 230, 180),
        (60, 60, 90),
    ]
    for idx, color in enumerate(colors):
        p = _make_textured_image(tmp_path / f"tile_{idx}.jpg", size=(200, 200), base_rgb=color)
        tile_paths.append(p)

    out_mosaic = tmp_path / "mosaic_a4.jpg"
    cfg = MosaicConfig(paper_size="A4", dpi=100, cols=12, max_reuse=30, min_distance=1)
    res = build_mosaic(target_path, tile_paths, out_mosaic, config=cfg)

    assert out_mosaic.exists()
    # A4 portrait at 100 DPI: round(21.0/2.54 * 100)=827, round(29.7/2.54 * 100)=1169
    assert res.width_px == 827
    assert res.height_px == 1169
    with Image.open(out_mosaic) as saved:
        assert saved.size == (827, 1169)
        assert round(saved.info["dpi"][0]) == 100


def test_video_render_with_audio_and_mosaic_finale(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(3):
        exif = _build_exif(dt_str=f"2025:07:0{i + 1} 10:00:00")
        size = (400, 600) if i == 1 else (640, 360)
        _make_textured_image(
            photos_dir / f"p_{i}.jpg",
            size=size,
            base_rgb=(70 + i * 50, 120, 190 - i * 40),
            exif_bytes=exif,
        )

    target_path = tmp_path / "target.jpg"
    _make_textured_image(target_path, size=(400, 560), base_rgb=(180, 140, 120))

    # Generate a 12-second sine-wave test audio file via ffmpeg
    audio_path = tmp_path / "song.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:a",
            "aac",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    manifest = scan_photos(photos_dir, IngestConfig(dedup_hash_distance=-1, min_sharpness=0.0))
    cluster_trips(manifest, TripConfig())

    mosaic_out = tmp_path / "mosaic.jpg"
    build_mosaic(
        target_path,
        [p.path for p in manifest.photos],
        mosaic_out,
        config=MosaicConfig(paper_size="A4", dpi=72, cols=8),
    )

    video_out = tmp_path / "keepsake.mp4"
    v_cfg = VideoConfig(
        resolution="720P",
        fps=15,
        seconds_per_photo=2.5,
        transition_sec=0.5,
        title_card_sec=1.5,
        mosaic_finale_sec=2.5,
        preset="ultrafast",
        crf=28,
    )
    v_res = render_keepsake_video(
        manifest,
        video_out,
        config=v_cfg,
        audio_paths=[audio_path],
        mosaic_image_path=mosaic_out,
    )

    assert video_out.exists()
    assert v_res.has_audio is True
    assert v_res.photo_count == 3
    dur = probe_media_duration(video_out)
    assert dur >= 7.0


def test_config_toml_and_cli_scan(tmp_path: Path) -> None:
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text(
        '[ingest]\nmax_video_photos = 10\nmin_sharpness = 0.0\n[mosaic]\npaper_size = "A3"\n',
        encoding="utf-8",
    )
    cfg = ProjectConfig.from_toml(toml_path)
    assert cfg.ingest.max_video_photos == 10
    assert cfg.mosaic.paper_size == "A3"

    photos_dir = tmp_path / "pics"
    photos_dir.mkdir()
    _make_textured_image(photos_dir / "a.jpg")
    manifest_path = tmp_path / "manifest.json"
    rc = cli_main(["--config", str(toml_path), "scan", str(photos_dir), "-o", str(manifest_path)])
    assert rc == 0
    assert manifest_path.exists()


def test_config_warns_on_unknown_keys(tmp_path: Path) -> None:
    toml_path = tmp_path / "bad_cfg.toml"
    toml_path.write_text(
        "[ingest]\nmax_video_photos = 12\ntypo_key = true\n[unknown_section]\nfoo = 1\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning) as caught:
        cfg = ProjectConfig.from_toml(toml_path)

    messages = "\n".join(str(w.message) for w in caught)
    assert "unknown_section" in messages
    assert "typo_key" in messages
    assert cfg.ingest.max_video_photos == 12


def test_raw_tiff_exif_and_parallel_scan(tmp_path: Path) -> None:
    exif = _build_exif(
        dt_str="2025:09:15 16:45:00",
        make="Leica",
        model="Q3",
        lat_dms=((48, 1), (8, 1), (0, 1)),
        lat_ref="N",
        lon_dms=((11, 1), (34, 1), (0, 1)),
        lon_ref="E",
    )
    dng_path = tmp_path / "sample.dng"
    Image.new("RGB", (64, 64), (100, 120, 140)).save(dng_path, format="TIFF", exif=exif)

    parsed = _read_exif_dict(dng_path)
    assert parsed["make"] == "Leica"
    assert parsed["model"] == "Q3"
    assert parsed["datetime_original"] == "2025:09:15 16:45:00"
    assert parsed["latitude"] == pytest.approx(48.133333, abs=1e-4)
    assert parsed["longitude"] == pytest.approx(11.566667, abs=1e-4)

    pics_dir = tmp_path / "pics"
    pics_dir.mkdir()
    for i in range(6):
        ex = _build_exif(dt_str=f"2025:09:0{i + 1} 10:00:00")
        _make_textured_image(
            pics_dir / f"img_{i}.jpg",
            base_rgb=(40 + i * 30, 100, 200 - i * 20),
            exif_bytes=ex,
        )

    seq = scan_photos(
        pics_dir, IngestConfig(min_sharpness=0.0, dedup_hash_distance=-1, scan_workers=1)
    )
    par = scan_photos(
        pics_dir, IngestConfig(min_sharpness=0.0, dedup_hash_distance=-1, scan_workers=4)
    )
    assert [p.path for p in seq.photos] == [p.path for p in par.photos]
    assert [p.sharpness for p in seq.photos] == [p.sharpness for p in par.photos]
    assert [p.dhash for p in seq.photos] == [p.dhash for p in par.photos]


def test_mosaic_zoom_reveal_custom_focal_point(tmp_path: Path) -> None:
    mosaic_img = Image.new("RGB", (400, 560), (20, 40, 180))
    draw = ImageDraw.Draw(mosaic_img)
    draw.rectangle([20, 20, 180, 180], fill=(240, 220, 60))
    mosaic_path = tmp_path / "mosaic.png"
    mosaic_img.save(mosaic_path)

    out_vid = tmp_path / "reveal.mp4"
    render_mosaic_zoom_reveal(
        mosaic_path,
        out_vid,
        width=320,
        height=180,
        duration_sec=1.0,
        fps=10,
        start_zoom=4.0,
        focal_x=0.2,
        focal_y=0.2,
        preset="ultrafast",
        crf=28,
    )
    assert out_vid.exists()
    assert probe_media_duration(out_vid) >= 0.9


def test_project_section_config_driven_run(tmp_path: Path) -> None:
    proj_dir = tmp_path / "my_project"
    photos_dir = proj_dir / "photos"
    audio_dir = proj_dir / "audio"
    photos_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    for i in range(2):
        exif = _build_exif(dt_str=f"2025:10:0{i + 1} 12:00:00")
        _make_textured_image(
            photos_dir / f"p_{i}.jpg",
            base_rgb=(80 + i * 70, 120, 180 - i * 40),
            exif_bytes=exif,
        )

    target_path = proj_dir / "portrait.jpg"
    _make_textured_image(target_path, size=(300, 420), base_rgb=(180, 140, 120))

    track_path = audio_dir / "track1.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-c:a",
            "aac",
            str(track_path),
        ],
        check=True,
        capture_output=True,
    )

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'photos_dir = "./photos"',
                'target_image = "./portrait.jpg"',
                'out_dir = "./out"',
                'audio_tracks = ["./audio/track1.m4a"]',
                "",
                "[ingest]",
                "min_sharpness = 0.0",
                "dedup_hash_distance = -1",
                "",
                "[mosaic]",
                'paper_size = "A4"',
                "dpi = 72",
                "cols = 6",
                "",
                "[video]",
                'resolution = "720P"',
                "fps = 12",
                "seconds_per_photo = 2.0",
                "transition_sec = 0.4",
                "title_card_sec = 1.0",
                "mosaic_finale_sec = 1.5",
                'preset = "ultrafast"',
                "crf = 28",
            ]
        ),
        encoding="utf-8",
    )

    rc = cli_main(["--config", str(toml_path), "run"])
    assert rc == 0
    assert (proj_dir / "out" / "manifest.json").exists()
    assert (proj_dir / "out" / "mosaic.jpg").exists()
    assert (proj_dir / "out" / "keepsake.mp4").exists()


def test_config_scan_and_mosaic_share_out_dir_manifest(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    photos_dir = proj_dir / "photos"
    photos_dir.mkdir(parents=True)
    _make_textured_image(photos_dir / "a.jpg", base_rgb=(100, 140, 180))
    target_path = proj_dir / "portrait.jpg"
    _make_textured_image(target_path, size=(240, 320), base_rgb=(120, 140, 160))

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'photos_dir = "./photos"',
                'target_image = "./portrait.jpg"',
                'out_dir = "./out"',
                "",
                "[ingest]",
                "min_sharpness = 0.0",
                "dedup_hash_distance = -1",
                "",
                "[mosaic]",
                'paper_size = "A4"',
                "dpi = 72",
                "cols = 6",
            ]
        ),
        encoding="utf-8",
    )

    assert cli_main(["--config", str(toml_path), "scan"]) == 0
    assert (proj_dir / "out" / "manifest.json").exists()
    assert cli_main(["--config", str(toml_path), "mosaic"]) == 0
    assert (proj_dir / "out" / "mosaic.jpg").exists()


def test_audio_climax_detection_slicing_and_beats(tmp_path: Path) -> None:
    assert parse_timestamp_sec("01:35.5") == pytest.approx(95.5)
    parsed = parse_audio_spec("song.mp3@01:10-01:50")
    assert parsed.path == Path("song.mp3")
    assert parsed.start_sec == pytest.approx(70.0)
    assert parsed.end_sec == pytest.approx(110.0)

    # 12s audio: quiet 0..4s, pulsed loud climax 4..8s, quiet 8..12s
    song_a = tmp_path / "song_a.wav"
    import wave

    sr = 16000
    t = np.arange(sr * 12, dtype=np.float32) / sr
    env = np.where((t >= 4.0) & (t < 8.0), 0.35 + 0.55 * ((t % 0.5) > 0.38), 0.05)
    freq = np.where((t >= 4.0) & (t < 8.0), 440.0, 220.0)
    pcm = (np.clip(env * np.sin(2.0 * np.pi * freq * t), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(song_a), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    start_sec, end_sec, peak_sec = detect_climax_window(song_a, target_duration_sec=4.0)
    assert 3.0 <= start_sec <= 5.0
    assert 7.0 <= end_sec <= 9.0
    assert start_sec <= peak_sec <= end_sec

    timeline = analyze_audio_dynamics(song_a, step_sec=0.1)
    assert len(timeline.energy) >= 110
    quiet_energy = float(np.mean(timeline.energy[:30]))
    climax_energy = float(np.mean(timeline.energy[45:75]))
    assert climax_energy > quiet_energy * 2.5
    assert any(3.8 <= bt <= 8.2 for bt in timeline.beat_times)


def test_audio_cli_climax_mix_and_toml_audio_clips(tmp_path: Path) -> None:
    proj_dir = tmp_path / "aud_proj"
    audio_dir = proj_dir / "audio"
    audio_dir.mkdir(parents=True)

    for idx, freq in enumerate((330, 520), start=1):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={freq}:duration=10",
                "-c:a",
                "aac",
                str(audio_dir / f"track_{idx}.m4a"),
            ],
            check=True,
            capture_output=True,
        )

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'out_dir = "./out"',
                "",
                "[[project.audio_clips]]",
                'path = "./audio/track_1.m4a"',
                'start = "00:01"',
                'end = "00:05"',
                "",
                "[[project.audio_clips]]",
                'path = "./audio/track_2.m4a"',
                "start = 2.0",
                "end = 6.0",
                "",
                "[video]",
                "audio_crossfade_sec = 1.0",
                "audio_fade_in_sec = 0.3",
                "audio_fade_out_sec = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    mix_out = proj_dir / "out" / "mix_preview.m4a"
    rc = cli_main(["--config", str(toml_path), "audio"])
    assert rc == 0
    assert mix_out.exists()
    dur = probe_media_duration(mix_out)
    # 4s + 4s - 1s crossfade = 7.0s (much shorter than 20s full tracks)
    assert 6.5 <= dur <= 7.5


def test_audio_dir_folder_discovery(tmp_path: Path) -> None:
    proj_dir = tmp_path / "dir_proj"
    audio_dir = proj_dir / "audio"
    audio_dir.mkdir(parents=True)

    for name in ("02_second.mp3", "01_first.m4a"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=4",
                str(audio_dir / name),
            ],
            check=True,
            capture_output=True,
        )
    # Non-audio and hidden files should be ignored
    (audio_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (audio_dir / ".hidden.mp3").write_bytes(b"ignore")

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'out_dir = "./out"',
                'audio_dir = "./audio"',
                "",
                "[video]",
                "audio_crossfade_sec = 1.0",
                "audio_fade_in_sec = 0.2",
                "audio_fade_out_sec = 0.3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = ProjectConfig.from_toml(toml_path)
    specs = cfg.project.resolved_audio_specs()
    assert [s.path.name for s in specs] == ["01_first.m4a", "02_second.mp3"]

    assert cli_main(["--config", str(toml_path), "audio"]) == 0
    mix_out = proj_dir / "out" / "mix_preview.m4a"
    assert mix_out.exists()
    assert 6.5 <= probe_media_duration(mix_out) <= 7.5


def test_audio_dir_with_per_clip_override(tmp_path: Path) -> None:
    proj_dir = tmp_path / "merge_proj"
    songs_dir = tmp_path / "external_songs"
    proj_dir.mkdir()
    songs_dir.mkdir()

    for name in ("a_song.mp3", "b_song.mp3"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6",
                str(songs_dir / name),
            ],
            check=True,
            capture_output=True,
        )

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                f'audio_dir = "{songs_dir}"',
                "",
                "[[project.audio_clips]]",
                'path = "a_song.mp3"',
                'start = "00:01"',
                'end = "00:04"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = ProjectConfig.from_toml(toml_path)
    specs = cfg.project.resolved_audio_specs()
    assert [s.path.name for s in specs] == ["a_song.mp3", "b_song.mp3"]
    assert specs[0].start_sec == pytest.approx(1.0)
    assert specs[0].end_sec == pytest.approx(4.0)
    assert specs[1].start_sec is None
    assert specs[1].end_sec is None


def test_dynamic_pacing_bursts_heroes_and_hybrid_render(tmp_path: Path) -> None:
    import wave

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    # 8 photos across two chapters, including a dense afternoon cluster
    timestamps = [
        "2025:06:01 09:00:00",  # anchor 1
        "2025:06:01 15:00:00",
        "2025:06:01 15:01:00",
        "2025:06:01 15:02:00",
        "2025:06:01 15:03:00",
        "2025:06:01 15:04:00",
        "2025:06:05 10:00:00",  # chapter 2 anchor
        "2025:06:05 18:00:00",  # final anchor
    ]
    for idx, ts in enumerate(timestamps):
        exif = _build_exif(dt_str=ts)
        _make_textured_image(
            photos_dir / f"p_{idx:02d}.jpg",
            size=(640, 360) if idx % 3 != 1 else (360, 540),
            base_rgb=(60 + idx * 22, 130, 210 - idx * 18),
            exif_bytes=exif,
        )

    # 12s audio: quiet 0..3s, intense pulsed climax 3..8s, gentle outro 8..12s
    wav_path = tmp_path / "dynamics.wav"
    sr = 16000
    t = np.arange(sr * 12, dtype=np.float32) / sr
    env = np.where(
        (t >= 3.0) & (t < 8.0),
        0.50 + 0.45 * ((t % 0.35) > 0.25),
        0.08,
    )
    pcm = (np.clip(env * np.sin(2.0 * np.pi * 440.0 * t), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    manifest = scan_photos(photos_dir, IngestConfig(min_sharpness=0.0, dedup_hash_distance=-1))
    cluster_trips(manifest, TripConfig(max_gap_hours=24.0))

    timeline = analyze_audio_dynamics(wav_path, step_sec=0.1)
    v_cfg = VideoConfig(
        resolution="720P",
        fps=15,
        pacing_mode="dynamic",
        min_clip_sec=0.20,
        max_clip_sec=3.20,
        transition_sec=0.60,
        include_title_cards=False,
        mosaic_finale_sec=0.0,
        preset="ultrafast",
        crf=28,
    )

    plan = plan_video_pacing(manifest, 12.0, v_cfg, timeline=timeline, persist_to_manifest=True)
    assert plan.min_clip_sec <= 0.65
    assert plan.max_clip_sec >= 1.5
    assert plan.burst_count >= 1
    assert plan.hero_count >= 1
    assert any(p.duration_sec is not None and p.pacing_reason for p in manifest.photos)

    # Render end-to-end mixing short hard-cut clips + longer crossfaded Ken Burns clips
    out_mp4 = tmp_path / "dynamic.mp4"
    res = render_keepsake_video(
        manifest,
        out_mp4,
        config=v_cfg,
        audio_paths=[wav_path],
    )
    assert out_mp4.exists()
    assert res.has_audio is True
    assert res.pacing_plan is not None
    dur = probe_media_duration(out_mp4)
    assert 9.5 <= dur <= 13.5


def test_cli_plan_command_persists_pacing(tmp_path: Path) -> None:
    proj_dir = tmp_path / "plan_proj"
    photos_dir = proj_dir / "photos"
    photos_dir.mkdir(parents=True)
    for idx in range(5):
        exif = _build_exif(dt_str=f"2025:07:01 12:0{idx}:00")
        _make_textured_image(
            photos_dir / f"img_{idx}.jpg",
            base_rgb=(70 + idx * 35, 110, 190 - idx * 25),
            exif_bytes=exif,
        )

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'photos_dir = "./photos"',
                'out_dir = "./out"',
                "",
                "[ingest]",
                "min_sharpness = 0.0",
                "dedup_hash_distance = -1",
                "",
                "[video]",
                "target_duration_sec = 8.0",
                "include_title_cards = false",
                'pacing_mode = "dynamic"',
            ]
        ),
        encoding="utf-8",
    )

    assert cli_main(["--config", str(toml_path), "scan"]) == 0
    assert cli_main(["--config", str(toml_path), "plan"]) == 0

    loaded = Manifest.load(proj_dir / "out" / "manifest.json")
    planned = [p for p in loaded.photos if p.selected_for_video]
    assert len(planned) == 5
    assert all(p.duration_sec is not None and p.video_role is not None for p in planned)


def test_config_auto_picks_mosaic_finale_for_video(tmp_path: Path) -> None:
    proj_dir = tmp_path / "auto_finale"
    photos_dir = proj_dir / "photos"
    photos_dir.mkdir(parents=True)
    for idx in range(2):
        exif = _build_exif(dt_str=f"2025:08:0{idx + 1} 12:00:00")
        _make_textured_image(
            photos_dir / f"p_{idx}.jpg",
            base_rgb=(90 + idx * 60, 120, 170),
            exif_bytes=exif,
        )

    target_path = proj_dir / "portrait.jpg"
    _make_textured_image(target_path, size=(240, 320), base_rgb=(150, 120, 110))

    toml_path = proj_dir / "keepsake.toml"
    toml_path.write_text(
        "\n".join(
            [
                "[project]",
                'photos_dir = "./photos"',
                'target_image = "./portrait.jpg"',
                'out_dir = "./out"',
                "",
                "[ingest]",
                "min_sharpness = 0.0",
                "dedup_hash_distance = -1",
                "",
                "[mosaic]",
                'paper_size = "A4"',
                "dpi = 72",
                "cols = 6",
                "",
                "[video]",
                'resolution = "720P"',
                "fps = 12",
                "seconds_per_photo = 1.5",
                "transition_sec = 0.4",
                "include_title_cards = false",
                "mosaic_finale_sec = 1.5",
                'preset = "ultrafast"',
                "crf = 28",
            ]
        ),
        encoding="utf-8",
    )

    assert cli_main(["--config", str(toml_path), "scan"]) == 0
    assert cli_main(["--config", str(toml_path), "mosaic"]) == 0
    assert (proj_dir / "out" / "mosaic.jpg").exists()

    # Zero path flags needed for `video` — auto-picks <out_dir>/mosaic.jpg as finale
    assert cli_main(["--config", str(toml_path), "video"]) == 0
    out_mp4 = proj_dir / "out" / "keepsake.mp4"
    assert out_mp4.exists()
    # 2 * 1.5s photos + 1.5s finale - overlaps >= 3.3s
    assert probe_media_duration(out_mp4) >= 3.2


def test_year_cards_closing_quote_and_mosaic_puzzle_finale(tmp_path: Path) -> None:
    import wave

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    # 6 photos across 2 years (2021 & 2022), 4 distinct trip clusters
    dates = [
        "2021:03:01 10:00:00",
        "2021:08:15 14:00:00",
        "2021:08:15 14:02:00",
        "2022:02:10 09:00:00",
        "2022:09:20 16:00:00",
        "2022:09:20 16:01:00",
    ]
    for idx, ts in enumerate(dates):
        exif = _build_exif(dt_str=ts)
        _make_textured_image(
            photos_dir / f"y_{idx:02d}.jpg",
            base_rgb=(50 + idx * 32, 120, 210 - idx * 25),
            exif_bytes=exif,
        )

    target_path = tmp_path / "portrait.jpg"
    _make_textured_image(target_path, size=(240, 320), base_rgb=(170, 130, 110))

    manifest = scan_photos(photos_dir, IngestConfig(min_sharpness=0.0, dedup_hash_distance=-1))
    cluster_trips(manifest, TripConfig(max_gap_hours=48.0))
    assert len(manifest.trips) == 4  # 4 trip clusters, but only 2 distinct years

    mosaic_out = tmp_path / "mosaic.jpg"
    build_mosaic(
        target_path,
        [p.path for p in manifest.photos],
        mosaic_out,
        config=MosaicConfig(paper_size="A4", dpi=72, cols=10),
    )
    assert mosaic_out.with_suffix(".grid.json").exists()

    # 12s dynamic audio with loud middle section
    wav_path = tmp_path / "track.wav"
    sr = 16000
    t = np.arange(sr * 12, dtype=np.float32) / sr
    env = np.where((t >= 2.5) & (t < 7.5), 0.75, 0.12)
    pcm = (np.clip(env * np.sin(2.0 * np.pi * 440.0 * t), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    v_cfg = VideoConfig(
        resolution="720P",
        fps=12,
        pacing_mode="dynamic",
        target_avg_clip_sec=0.75,
        min_clip_sec=0.20,
        max_clip_sec=2.0,
        motion_sync=True,
        title_card_mode="year",
        year_card_prefix="Year ",
        title_card_sec=1.2,
        closing_quote="memories worth keeping",
        closing_quote_sec=1.5,
        mosaic_finale_sec=2.5,
        mosaic_finale_style="puzzle",
        mosaic_puzzle_pattern="fly_in",
        transition_sec=0.4,
        preset="ultrafast",
        crf=28,
    )

    out_mp4 = tmp_path / "keepsake_puzzle.mp4"
    res = render_keepsake_video(
        manifest,
        out_mp4,
        config=v_cfg,
        audio_paths=[wav_path],
        mosaic_image_path=mosaic_out,
        mosaic_cols=10,
    )
    assert out_mp4.exists()
    assert res.chapter_count == 2  # Year 2021 + Year 2022
    assert res.has_audio is True
    vid_dur = probe_media_duration(out_mp4)
    aud_dur = probe_media_duration(wav_path)
    assert abs(vid_dur - aud_dur) <= 0.45

    # Verify MP4 has moov atom before mdat (+faststart for iCloud / QuickTime streaming)
    atoms: list[str] = []
    with out_mp4.open("rb") as fh:
        pos = 0
        size = out_mp4.stat().st_size
        while pos < size:
            fh.seek(pos)
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            sz = int.from_bytes(hdr[:4], "big")
            typ = hdr[4:8].decode("ascii", errors="replace")
            if sz == 1:
                sz = int.from_bytes(fh.read(8), "big")
            atoms.append(typ)
            if sz <= 0:
                break
            pos += sz
    assert atoms.index("moov") < atoms.index("mdat")

    # Also test center_out puzzle pattern directly
    puzzle_out = tmp_path / "puzzle_center.mp4"
    render_mosaic_puzzle_finale(
        mosaic_out,
        puzzle_out,
        width=320,
        height=180,
        duration_sec=1.2,
        fps=10,
        pattern="center_out",
        preset="ultrafast",
        crf=28,
    )
    assert puzzle_out.exists()
    assert probe_media_duration(puzzle_out) >= 1.0


def test_example_toml_parses_cleanly_without_warnings() -> None:
    import warnings

    example_path = Path(__file__).resolve().parent.parent / "keepsake.example.toml"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = ProjectConfig.from_toml(example_path)
    assert caught == []
    assert cfg.video.title_card_mode == "year"
    assert cfg.video.mosaic_finale_style == "puzzle"
    assert cfg.video.render_workers == 0


def test_logger_colors_levels_and_no_duplicates() -> None:
    import io

    buf = io.StringIO()
    configure_logging(stream=buf, force_color=True)
    # Reconfiguring should replace handler rather than duplicating output
    configure_logging(stream=buf, force_color=True)
    log = get_logger("test_mod")
    log.debug("hidden debug")
    log.info("hello info")
    log.warning("careful warning")

    colored = buf.getvalue()
    assert "hidden debug" not in colored
    assert colored.count("hello info") == 1
    assert "careful warning" in colored
    assert "\033[36m" in colored
    assert "\033[33;1m" in colored

    plain_buf = io.StringIO()
    configure_logging(stream=plain_buf, quiet=True, force_color=False)
    log.info("suppressed info")
    log.warning("visible warning")
    plain = plain_buf.getvalue()
    assert "suppressed info" not in plain
    assert "visible warning" in plain
    assert "\033[" not in plain


def test_parallel_video_render_and_progress_logging(tmp_path: Path) -> None:
    import io

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for idx in range(4):
        exif = _build_exif(dt_str=f"2024:06:0{idx + 1} 12:00:00")
        _make_textured_image(
            photos_dir / f"p_{idx:02d}.jpg",
            base_rgb=(60 + idx * 45, 130, 200 - idx * 30),
            exif_bytes=exif,
        )

    manifest = scan_photos(photos_dir, IngestConfig(min_sharpness=0.0, dedup_hash_distance=-1))
    cluster_trips(manifest, TripConfig(max_gap_hours=72.0))

    common = dict(
        resolution="720P",
        fps=12,
        seconds_per_photo=1.0,
        transition_sec=0.3,
        include_title_cards=True,
        title_card_mode="year",
        title_card_sec=1.0,
        closing_quote="a brief closing line",
        closing_quote_sec=1.0,
        mosaic_finale_sec=0.0,
        preset="ultrafast",
        crf=28,
    )

    seq_out = tmp_path / "seq.mp4"
    res_seq = render_keepsake_video(
        manifest,
        seq_out,
        config=VideoConfig(**common, render_workers=1),
    )

    log_buf = io.StringIO()
    configure_logging(stream=log_buf, force_color=False)
    par_out = tmp_path / "par.mp4"
    res_par = render_keepsake_video(
        manifest,
        par_out,
        config=VideoConfig(**common, render_workers=2),
    )

    assert seq_out.exists()
    assert par_out.exists()
    assert res_par.photo_count == res_seq.photo_count == 4
    assert res_par.chapter_count == res_seq.chapter_count == 1
    assert abs(probe_media_duration(par_out) - probe_media_duration(seq_out)) <= 0.15

    logs = log_buf.getvalue()
    assert "Rendering 6 clip(s)" in logs
    assert "(2 worker(s))" in logs
    assert "Rendered 6/6 clips (100%)" in logs
    assert "Joining 6 rendered clip(s)..." in logs
    assert "Finalizing faststart MP4..." in logs
