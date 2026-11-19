"""EXIF metadata extraction, camera clock offset calibration, sharpness, and dhash."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from keepsake._image_io import RAW_EXTENSIONS, open_image, read_raw_exif_image
from keepsake._logging import get_logger
from keepsake._models import PhotoRecord

logger = get_logger(__name__)

_EXIF_DT_FORMAT = "%Y:%m:%d %H:%M:%S"
_EXIF_IFD = ExifTags.IFD.Exif
_GPS_IFD = ExifTags.IFD.GPSInfo

# Standard EXIF tag numeric IDs
_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_DATETIME = 306
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868

# GPS sub-IFD numeric IDs
_GPS_LAT_REF = 1
_GPS_LAT = 2
_GPS_LON_REF = 3
_GPS_LON = 4


def extract_photo_record(
    path: str | Path,
    camera_time_offsets_hours: dict[str, float] | None = None,
) -> PhotoRecord:
    """Inspect a photo file and return its normalized metadata and quality metrics."""
    file_path = Path(path).resolve()
    exif_data = _read_exif_dict(file_path)

    make = _clean_str(exif_data.get("make"))
    model = _clean_str(exif_data.get("model"))
    raw_dt, dt_source = _resolve_timestamp(file_path, exif_data)

    if raw_dt is not None and camera_time_offsets_hours:
        offset_h = _match_camera_offset(make, model, camera_time_offsets_hours)
        if offset_h:
            raw_dt = raw_dt + timedelta(hours=offset_h)

    with open_image(file_path, mode="RGB") as img:
        width, height = img.size
        sharpness = compute_sharpness(img)
        dhash = compute_dhash(img)

    return PhotoRecord(
        path=str(file_path),
        timestamp=raw_dt.isoformat() if raw_dt else None,
        timestamp_source=dt_source,
        latitude=exif_data.get("latitude"),
        longitude=exif_data.get("longitude"),
        camera_make=make,
        camera_model=model,
        width=width,
        height=height,
        sharpness=round(sharpness, 2),
        dhash=dhash,
    )


def compute_sharpness(img: Image.Image, max_edge: int = 800) -> float:
    """Estimate image focus sharpness using 2D Laplacian variance on grayscale pixels."""
    gray = img.convert("L")
    if max(gray.size) > max_edge:
        gray.thumbnail((max_edge, max_edge), Image.Resampling.BILINEAR)

    arr = np.asarray(gray, dtype=np.float32)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0

    laplacian = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4.0 * arr[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit difference perceptual hash encoded as a 16-char hex string."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    hex_len = (hash_size * hash_size) // 4
    return f"{value:0{hex_len}x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """Return bit-level Hamming distance between two hex-encoded perceptual hashes."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def _read_exif_dict(path: Path) -> dict[str, Any]:
    img = read_raw_exif_image(path)
    if img is not None:
        try:
            with img:
                exif = img.getexif()
                if exif:
                    exif_ifd = exif.get_ifd(_EXIF_IFD)
                    gps_ifd = exif.get_ifd(_GPS_IFD)

                    lat = _parse_gps_coord(gps_ifd.get(_GPS_LAT), gps_ifd.get(_GPS_LAT_REF))
                    lon = _parse_gps_coord(gps_ifd.get(_GPS_LON), gps_ifd.get(_GPS_LON_REF))

                    data = {
                        "make": exif.get(_TAG_MAKE),
                        "model": exif.get(_TAG_MODEL),
                        "datetime": exif.get(_TAG_DATETIME),
                        "datetime_original": exif_ifd.get(_TAG_DATETIME_ORIGINAL),
                        "datetime_digitized": exif_ifd.get(_TAG_DATETIME_DIGITIZED),
                        "latitude": lat,
                        "longitude": lon,
                    }
                    if any(v is not None for v in data.values()):
                        return data
        except Exception:
            pass

    if path.suffix.lower() in RAW_EXTENSIONS:
        return _read_exiftool_dict(path)
    return {}


def _read_exiftool_dict(path: Path) -> dict[str, Any]:
    exiftool_bin = shutil.which("exiftool")
    if exiftool_bin is None:
        return {}
    cmd = [
        exiftool_bin,
        "-json",
        "-n",
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
        "-Make",
        "-Model",
        "-GPSLatitude",
        "-GPSLongitude",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return {}
        payload = json.loads(res.stdout)
        if not payload:
            return {}
        entry = payload[0]
        lat = entry.get("GPSLatitude")
        lon = entry.get("GPSLongitude")
        return {
            "make": entry.get("Make"),
            "model": entry.get("Model"),
            "datetime": entry.get("ModifyDate"),
            "datetime_original": entry.get("DateTimeOriginal"),
            "datetime_digitized": entry.get("CreateDate"),
            "latitude": round(float(lat), 6) if lat is not None else None,
            "longitude": round(float(lon), 6) if lon is not None else None,
        }
    except Exception:
        return {}


def _resolve_timestamp(path: Path, exif_data: dict[str, Any]) -> tuple[datetime | None, str]:
    candidates = (
        ("exif:DateTimeOriginal", exif_data.get("datetime_original")),
        ("exif:DateTimeDigitized", exif_data.get("datetime_digitized")),
        ("exif:DateTime", exif_data.get("datetime")),
    )
    for source, raw_val in candidates:
        parsed = _parse_exif_datetime(raw_val)
        if parsed is not None:
            return parsed, source

    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime), "file:mtime"
    except OSError:
        return None, "none"


def _parse_exif_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().split("\x00")[0]
    try:
        return datetime.strptime(cleaned, _EXIF_DT_FORMAT)
    except ValueError:
        return None


def _parse_gps_coord(dms: Any, ref: Any) -> float | None:
    if not dms or len(dms) != 3:
        return None
    try:
        deg, mins, secs = (_rational_to_float(part) for part in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    decimal = deg + mins / 60.0 + secs / 3600.0
    ref_str = str(ref).strip().upper() if ref else ""
    if ref_str in {"S", "W"}:
        decimal = -abs(decimal)
    elif ref_str in {"N", "E"}:
        decimal = abs(decimal)
    return round(decimal, 6)


def _rational_to_float(val: Any) -> float:
    if isinstance(val, tuple) and len(val) == 2:
        num, den = val
        return float(num) / float(den)
    return float(val)


def _match_camera_offset(
    make: str | None,
    model: str | None,
    offsets: dict[str, float],
) -> float:
    haystack = " ".join(part for part in (make, model) if part).lower()
    if not haystack:
        return 0.0
    for pattern, hours in offsets.items():
        if pattern.lower() in haystack:
            return hours
    return 0.0


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    text = str(val).replace("\x00", "").strip()
    return text or None
