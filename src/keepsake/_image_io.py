"""Unified image loading across standard formats, HEIC/HEIF, and optional DSLR RAW."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import cast

from PIL import Image, ImageOps

RAW_EXTENSIONS = frozenset({".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf"})

_HEIF_LOCK = threading.Lock()
_HEIF_REGISTERED = False


def ensure_heif_support() -> bool:
    """Register pillow-heif opener once if installed."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return True
    with _HEIF_LOCK:
        if _HEIF_REGISTERED:
            return True
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            _HEIF_REGISTERED = True
            return True
        except ImportError:
            return False


def open_image(path: str | Path, *, mode: str | None = "RGB") -> Image.Image:
    """Open an image file, apply EXIF orientation, and optionally convert color mode."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in RAW_EXTENSIONS:
        img = _open_raw(p)
    else:
        ensure_heif_support()
        with Image.open(p) as raw_img:
            transposed = ImageOps.exif_transpose(raw_img)
            img = transposed.copy()

    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def read_raw_exif_image(path: str | Path) -> Image.Image | None:
    """Open an image without pixel decoding when only EXIF tags are needed."""
    p = Path(path)
    if p.suffix.lower() not in RAW_EXTENSIONS:
        ensure_heif_support()
    try:
        return Image.open(p)
    except Exception:
        return None


def _open_raw(path: Path) -> Image.Image:
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot decode RAW file '{path.name}' without 'rawpy'. "
            "Install optional RAW support with: uv sync --extra raw"
        ) from exc

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True)
    return cast(Image.Image, Image.fromarray(rgb))
