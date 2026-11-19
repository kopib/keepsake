# keepsake Examples

Runnable, self-contained examples that let you test the entire `keepsake` pipeline—photo curation, climax audio slicing, beat-synced variable video pacing, photomosaic portrait generation, and the live mosaic puzzle finale—without needing your own photos or music.

---

## Canada Landscape Demo (`examples/canada/`)

Builds a **~20-second 720p beat-synced highlight video** and a **landscape photomosaic of Moraine Lake (Banff, Canada)** from 50 **CC0 / Public Domain** Canada photos and a **CC0** audio track fetched on demand from Wikimedia Commons.

### 1. Fetch the CC0 / Public Domain demo inputs

```bash
uv run python examples/canada/fetch_demo.py
```

This downloads into `examples/canada/inputs/` (gitignored so the repository stays tiny):
- `inputs/target.jpg` — Moraine Lake, Banff (Public Domain)
- `inputs/photos/canada_001.jpg` … `canada_050.jpg` — 50 CC0 Canada landscape & city photos (800px thumbnails)
- `inputs/audio/01_theme.ogg` — CC0 instrumental soundtrack (`Monplaisir — Theme Song`)
- `inputs/SOURCES.md` — exact Wikimedia Commons titles and verified licenses

### 2. Run the full keepsake pipeline

```bash
uv run keepsake --config examples/canada/keepsake.toml run
```

Outputs are written to `examples/canada/output/` (gitignored):
- `manifest.json` — scanned EXIF metadata, sharpness scores, burst tags, and planned per-clip durations
- `mosaic.jpg` (+ `mosaic.grid.json`) — 42-column CIE Lab photomosaic of Moraine Lake
- `keepsake.mp4` — ~20s 720p beat-synced video with calm centered zoom, `"made with keepsake"` quote card, and a 7-second live mosaic puzzle finale

### 3. Try tweaking individual stages

Because every stage is decoupled through `manifest.json`, you can edit [`examples/canada/keepsake.toml`](canada/keepsake.toml) and rerun just one step:

```bash
# Try a different mosaic grid density or blend
uv run keepsake --config examples/canada/keepsake.toml mosaic --cols 52 --overlay-alpha 0.18

# Try the center-out or spiral puzzle pattern and re-render only the video
uv run keepsake --config examples/canada/keepsake.toml video --replan
```
