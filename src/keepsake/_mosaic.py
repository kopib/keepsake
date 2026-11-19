"""Print-ready photomosaic builder with CIE Lab sub-grid matching and zoom-out reveal."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree

from keepsake._config import MosaicConfig
from keepsake._image_io import open_image
from keepsake._logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MosaicResult:
    """Summary of a generated photomosaic image."""

    output_path: Path
    width_px: int
    height_px: int
    cols: int
    rows: int
    tile_width_px: int
    tile_height_px: int
    unique_tiles_used: int
    total_cells: int
    dpi: int


def build_mosaic(
    target_image_path: str | Path,
    tile_paths: list[str | Path],
    output_path: str | Path,
    config: MosaicConfig | None = None,
) -> MosaicResult:
    """Build a print-ready photomosaic from `tile_paths` matching `target_image_path`."""
    if not tile_paths:
        raise ValueError("tile_paths must contain at least one image.")

    cfg = config or MosaicConfig()
    with open_image(target_image_path, mode="RGB") as target_raw:
        canvas_w, canvas_h = cfg.resolved_canvas_px(target_raw.width, target_raw.height)
        cols = max(4, cfg.cols)
        tile_w = max(16, canvas_w // cols)
        rows = max(4, round(canvas_h / tile_w))
        tile_h = max(16, canvas_h // rows)
        grid_w, grid_h = cols * tile_w, rows * tile_h

        target_fitted = ImageOps.fit(target_raw, (grid_w, grid_h), method=Image.Resampling.LANCZOS)

    tiles_rgb, tile_features = _prepare_tile_bank(
        tile_paths, tile_w, tile_h, subgrid=max(1, cfg.subgrid)
    )
    if not tiles_rgb:
        raise ValueError("No readable tile images were found.")

    cell_features = _extract_target_cell_features(
        target_fitted,
        cols=cols,
        rows=rows,
        tile_w=tile_w,
        tile_h=tile_h,
        subgrid=max(1, cfg.subgrid),
    )

    assignments = _assign_tiles(
        cell_features,
        tile_features,
        cols=cols,
        rows=rows,
        max_reuse=max(1, cfg.max_reuse),
        min_distance=max(0, cfg.min_distance),
        seed=cfg.seed,
    )

    mosaic_img = Image.new("RGB", (grid_w, grid_h))
    for cell_idx, tile_idx in enumerate(assignments):
        r, c = divmod(cell_idx, cols)
        mosaic_img.paste(tiles_rgb[tile_idx], (c * tile_w, r * tile_h))

    if 0.0 < cfg.overlay_alpha <= 1.0:
        mosaic_img = Image.blend(mosaic_img, target_fitted, alpha=cfg.overlay_alpha)

    if (grid_w, grid_h) != (canvas_w, canvas_h):
        mosaic_img = mosaic_img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mosaic_img.save(out, dpi=(cfg.dpi, cfg.dpi), quality=95)
    grid_meta_path = out.with_suffix(".grid.json")
    grid_meta_path.write_text(
        json.dumps(
            {
                "cols": cols,
                "rows": rows,
                "tile_width_px": tile_w,
                "tile_height_px": tile_h,
                "width_px": canvas_w,
                "height_px": canvas_h,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return MosaicResult(
        output_path=out,
        width_px=canvas_w,
        height_px=canvas_h,
        cols=cols,
        rows=rows,
        tile_width_px=tile_w,
        tile_height_px=tile_h,
        unique_tiles_used=len(set(assignments)),
        total_cells=cols * rows,
        dpi=cfg.dpi,
    )


def render_mosaic_zoom_reveal(
    mosaic_image_path: str | Path,
    output_video_path: str | Path,
    *,
    width: int = 1920,
    height: int = 1080,
    duration_sec: float = 8.0,
    fps: int = 30,
    start_zoom: float = 8.0,
    focal_x: float = 0.5,
    focal_y: float = 0.5,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Render a smooth zoom-out video clip revealing the full mosaic from a focal tile."""
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fx_norm = min(1.0, max(0.0, focal_x))
    fy_norm = min(1.0, max(0.0, focal_y))

    with open_image(mosaic_image_path, mode="RGB") as mosaic:
        # High-res working frame on a dark slate canvas matching the target video aspect ratio
        hi_w, hi_h = width * 2, height * 2
        hi_mosaic = Image.new("RGB", (hi_w, hi_h), (14, 16, 20))
        hi_fitted = ImageOps.contain(mosaic, (hi_w, hi_h), method=Image.Resampling.LANCZOS)
        off_x = (hi_w - hi_fitted.width) // 2
        off_y = (hi_h - hi_fitted.height) // 2
        hi_mosaic.paste(hi_fitted, (off_x, off_y))

    start_cx = off_x + fx_norm * hi_fitted.width
    start_cy = off_y + fy_norm * hi_fitted.height
    end_cx, end_cy = hi_w / 2.0, hi_h / 2.0

    total_frames = max(2, round(duration_sec * fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame_idx in range(total_frames):
            t = frame_idx / (total_frames - 1)
            # Smooth ease-out cubic from start_zoom -> 1.0
            eased = 1.0 - (1.0 - t) ** 3
            zoom = start_zoom - (start_zoom - 1.0) * eased

            crop_w = max(2, round(hi_w / zoom))
            crop_h = max(2, round(hi_h / zoom))
            cx = start_cx + (end_cx - start_cx) * eased
            cy = start_cy + (end_cy - start_cy) * eased
            x0 = max(0, min(hi_w - crop_w, round(cx - crop_w / 2.0)))
            y0 = max(0, min(hi_h - crop_h, round(cy - crop_h / 2.0)))
            frame = hi_mosaic.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
                (width, height), Image.Resampling.BICUBIC
            )
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg zoom reveal failed ({ret}): {stderr[-500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()

    return out_path


def render_mosaic_puzzle_finale(
    mosaic_image_path: str | Path,
    output_video_path: str | Path,
    *,
    width: int = 1920,
    height: int = 1080,
    duration_sec: float = 12.0,
    fps: int = 30,
    cols: int | None = None,
    rows: int | None = None,
    pattern: str = "fly_in",
    focal_x: float = 0.5,
    focal_y: float = 0.5,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    seed: int = 42,
) -> Path:
    """Render a live puzzle-assembly animation constructing the mosaic from all its tiles."""
    mosaic_path = Path(mosaic_image_path)
    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read exact grid dimensions from .grid.json sidecar if present
    grid_cols, grid_rows = cols, rows
    sidecar = mosaic_path.with_suffix(".grid.json")
    if sidecar.exists() and (grid_cols is None or grid_rows is None):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            grid_cols = grid_cols or int(meta["cols"])
            grid_rows = grid_rows or int(meta["rows"])
        except Exception as exc:
            logger.warning("Could not read mosaic grid sidecar %s: %s", sidecar, exc)

    bg_color = np.array([12, 14, 19], dtype=np.uint8)
    with open_image(mosaic_path, mode="RGB") as mosaic:
        max_w = max(64, round(width * 0.94))
        max_h = max(64, round(height * 0.94))
        fitted = ImageOps.contain(mosaic, (max_w, max_h), method=Image.Resampling.LANCZOS)
        fit_w, fit_h = fitted.size
        mosaic_arr = np.asarray(fitted, dtype=np.uint8)

    if grid_cols is None:
        grid_cols = 48
    grid_cols = max(6, min(grid_cols, fit_w // 4))
    if grid_rows is None:
        grid_rows = max(6, round(grid_cols * fit_h / max(1, fit_w)))
    grid_rows = max(6, min(grid_rows, fit_h // 4))

    off_x = (width - fit_w) // 2
    off_y = (height - fit_h) // 2

    x_edges = np.round(np.linspace(0, fit_w, grid_cols + 1)).astype(np.int32)
    y_edges = np.round(np.linspace(0, fit_h, grid_rows + 1)).astype(np.int32)

    rng = np.random.default_rng(seed)
    fx_norm = min(1.0, max(0.0, focal_x))
    fy_norm = min(1.0, max(0.0, focal_y))
    cx_grid = fx_norm * (grid_cols - 1)
    cy_grid = fy_norm * (grid_rows - 1)
    max_rad = max(
        1.0,
        math.hypot(max(cx_grid, grid_cols - 1 - cx_grid), max(cy_grid, grid_rows - 1 - cy_grid)),
    )

    # Build per-tile schedule: (t_start, t_end, dst_x, dst_y, src_x, src_y, patch)
    t_assemble = max(0.8, duration_sec * 0.78)
    flight_dur = max(0.32, min(0.85, t_assemble * 0.14))
    max_start = max(0.05, t_assemble - flight_dur)
    fly_dist = max(width, height) * 0.28

    raw_tiles: list[tuple[float, int, int, int, int, np.ndarray]] = []
    for r in range(grid_rows):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        if y1 <= y0:
            continue
        for c in range(grid_cols):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if x1 <= x0:
                continue
            dx = c - cx_grid
            dy = r - cy_grid
            rad = math.hypot(dx, dy) / max_rad
            angle = (math.atan2(dy, dx) + math.pi) / (2.0 * math.pi)

            if pattern == "center_out":
                order_u = 0.78 * rad + 0.22 * float(rng.random())
            elif pattern == "spiral":
                order_u = 0.65 * rad + 0.25 * angle + 0.10 * float(rng.random())
            else:
                # "fly_in": outer frame + scattered puzzle pieces build organically into center
                order_u = 0.45 * (1.0 - rad) + 0.55 * float(rng.random())

            dst_x = off_x + x0
            dst_y = off_y + y0

            # Radial outward launch origin with slight angular swirl
            dir_ang = math.atan2(dy, dx) + float(rng.uniform(-0.35, 0.35))
            dist = fly_dist * float(rng.uniform(0.55, 1.35))
            src_x = round(dst_x + math.cos(dir_ang) * dist)
            src_y = round(dst_y + math.sin(dir_ang) * dist)
            patch = mosaic_arr[y0:y1, x0:x1]
            raw_tiles.append((order_u, dst_x, dst_y, src_x, src_y, patch))

    # Normalize order_u cleanly to [0, 1] and apply accelerating puzzle cascade curve (u^0.55)
    raw_tiles.sort(key=lambda item: item[0])
    n_tiles = max(1, len(raw_tiles))
    scheduled: list[tuple[float, float, int, int, int, int, np.ndarray]] = []
    for idx, (_, dst_x, dst_y, src_x, src_y, patch) in enumerate(raw_tiles):
        u = idx / max(1, n_tiles - 1)
        t_start = max_start * (u**0.55)
        t_end = t_start + flight_dur
        scheduled.append((t_start, t_end, dst_x, dst_y, src_x, src_y, patch))

    # Pre-sort by t_end for O(1) incremental stamping onto the settled canvas
    scheduled.sort(key=lambda item: item[1])
    settled = np.broadcast_to(bg_color, (height, width, 3)).copy()
    # Pre-build completed frame for crisp subtle push-in during the final reveal hold
    completed_canvas = settled.copy()
    completed_canvas[off_y : off_y + fit_h, off_x : off_x + fit_w] = mosaic_arr
    completed_img = Image.fromarray(completed_canvas, mode="RGB")

    total_frames = max(2, round(duration_sec * fps))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
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

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    settled_cursor = 0
    try:
        for frame_idx in range(total_frames):
            t = frame_idx / float(fps)

            while settled_cursor < len(scheduled) and scheduled[settled_cursor][1] <= t:
                _, _, dx, dy, _, _, patch = scheduled[settled_cursor]
                ph, pw = patch.shape[:2]
                settled[dy : dy + ph, dx : dx + pw] = patch
                settled_cursor += 1

            if settled_cursor >= len(scheduled):
                # Final completed hold with subtle 3% push-in
                hold_span = max(0.01, duration_sec - t_assemble)
                hold_u = min(1.0, max(0.0, (t - t_assemble) / hold_span))
                zoom = 1.0 + 0.03 * (1.0 - (1.0 - hold_u) ** 2)
                if zoom <= 1.001:
                    proc.stdin.write(completed_canvas.tobytes())
                else:
                    cw = max(2, round(width / zoom))
                    ch = max(2, round(height / zoom))
                    cx0 = (width - cw) // 2
                    cy0 = (height - ch) // 2
                    zoomed = completed_img.crop((cx0, cy0, cx0 + cw, cy0 + ch)).resize(
                        (width, height), Image.Resampling.BICUBIC
                    )
                    proc.stdin.write(zoomed.tobytes())
                continue

            frame = settled.copy()
            # Composite only tiles currently in flight (sorted by t_start/t_end)
            for idx in range(settled_cursor, len(scheduled)):
                t_start, t_end, dst_x, dst_y, src_x, src_y, patch = scheduled[idx]
                if t_start > t:
                    break
                p = min(1.0, max(0.0, (t - t_start) / max(1e-4, t_end - t_start)))
                eased = 1.0 - (1.0 - p) ** 3
                cur_x = round(src_x + (dst_x - src_x) * eased)
                cur_y = round(src_y + (dst_y - src_y) * eased)
                ph, pw = patch.shape[:2]

                # Clip cleanly to frame bounds
                fx0 = max(0, cur_x)
                fy0 = max(0, cur_y)
                fx1 = min(width, cur_x + pw)
                fy1 = min(height, cur_y + ph)
                if fx1 <= fx0 or fy1 <= fy0:
                    continue

                px0 = fx0 - cur_x
                py0 = fy0 - cur_y
                sub_patch = patch[py0 : py0 + (fy1 - fy0), px0 : px0 + (fx1 - fx0)]
                alpha = min(1.0, p * 3.0)
                if alpha >= 0.99:
                    frame[fy0:fy1, fx0:fx1] = sub_patch
                else:
                    dst_slice = frame[fy0:fy1, fx0:fx1].astype(np.float32)
                    blended = dst_slice * (1.0 - alpha) + sub_patch.astype(np.float32) * alpha
                    frame[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)

            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg puzzle finale failed ({ret}): {stderr[-500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()

    return out_path


def _prepare_tile_bank(
    tile_paths: list[str | Path],
    tile_w: int,
    tile_h: int,
    *,
    subgrid: int,
) -> tuple[list[Image.Image], np.ndarray]:
    tiles: list[Image.Image] = []
    features: list[np.ndarray] = []

    for path in tile_paths:
        try:
            with open_image(path, mode="RGB") as img:
                tile = ImageOps.fit(img, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        except Exception as exc:
            logger.warning("Skipping unreadable mosaic tile %s: %s", path, exc)
            continue
        tiles.append(tile)
        features.append(_image_subgrid_lab(tile, subgrid=subgrid))

    if not features:
        return [], np.empty((0, subgrid * subgrid * 3), dtype=np.float32)
    return tiles, np.vstack(features)


def _extract_target_cell_features(
    target: Image.Image,
    *,
    cols: int,
    rows: int,
    tile_w: int,
    tile_h: int,
    subgrid: int,
) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32)
    # Reshape (rows * tile_h, cols * tile_w, 3) -> per-cell subgrid means
    cell_feats: list[np.ndarray] = []
    for r in range(rows):
        y0, y1 = r * tile_h, (r + 1) * tile_h
        for c in range(cols):
            x0, x1 = c * tile_w, (c + 1) * tile_w
            patch = Image.fromarray(arr[y0:y1, x0:x1].astype(np.uint8))
            cell_feats.append(_image_subgrid_lab(patch, subgrid=subgrid))
    return np.vstack(cell_feats)


def _image_subgrid_lab(img: Image.Image, *, subgrid: int) -> np.ndarray:
    small = img.resize((subgrid, subgrid), Image.Resampling.BOX)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    lab = _srgb_to_lab(rgb)
    return lab.reshape(-1)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB [0, 1] array (..., 3) to perceptual CIE Lab."""
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    # sRGB D65 -> XYZ
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ m.T
    xyz[..., 0] /= 0.95047
    xyz[..., 2] /= 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _assign_tiles(
    cell_features: np.ndarray,
    tile_features: np.ndarray,
    *,
    cols: int,
    rows: int,
    max_reuse: int,
    min_distance: int,
    seed: int,
) -> list[int]:
    """Assign tile indices to cells center-outward while respecting reuse & spacing caps."""
    n_cells = cols * rows
    n_tiles = len(tile_features)
    k = min(n_tiles, max(24, max_reuse * 4))

    tree = cKDTree(tile_features)
    _, candidate_matrix = tree.query(cell_features, k=k)
    if k == 1:
        candidate_matrix = candidate_matrix[:, np.newaxis]

    # Process center cells first (most important for portrait faces) with light jitter
    rng = np.random.default_rng(seed)
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    order = sorted(
        range(n_cells),
        key=lambda idx: (
            math.hypot((idx // cols) - cy, (idx % cols) - cx) + float(rng.uniform(0.0, 0.75))
        ),
    )

    assignments = [-1] * n_cells
    usage_counts = [0] * n_tiles
    placed_coords: dict[int, list[tuple[int, int]]] = {}

    # Effective reuse cap expands automatically if photo pool is smaller than grid
    effective_max_reuse = max(max_reuse, math.ceil(n_cells / n_tiles))

    for cell_idx in order:
        r, c = divmod(cell_idx, cols)
        chosen = _pick_candidate(
            candidate_matrix[cell_idx],
            r=r,
            c=c,
            usage_counts=usage_counts,
            placed_coords=placed_coords,
            max_reuse=effective_max_reuse,
            min_distance=min_distance,
        )
        if chosen is None:
            # Relax distance first, then reuse cap
            chosen = _pick_candidate(
                candidate_matrix[cell_idx],
                r=r,
                c=c,
                usage_counts=usage_counts,
                placed_coords=placed_coords,
                max_reuse=effective_max_reuse,
                min_distance=0,
            )
        if chosen is None:
            chosen = int(
                min(
                    range(n_tiles),
                    key=lambda tidx: (
                        usage_counts[tidx],
                        float(np.linalg.norm(cell_features[cell_idx] - tile_features[tidx])),
                    ),
                )
            )

        assignments[cell_idx] = chosen
        usage_counts[chosen] += 1
        placed_coords.setdefault(chosen, []).append((r, c))

    return assignments


def _pick_candidate(
    candidates: np.ndarray,
    *,
    r: int,
    c: int,
    usage_counts: list[int],
    placed_coords: dict[int, list[tuple[int, int]]],
    max_reuse: int,
    min_distance: int,
) -> int | None:
    for raw_idx in candidates:
        tidx = int(raw_idx)
        if usage_counts[tidx] >= max_reuse:
            continue
        if min_distance > 0 and any(
            max(abs(r - pr), abs(c - pc)) < min_distance for pr, pc in placed_coords.get(tidx, ())
        ):
            continue
        return tidx
    return None
