"""Fetch CC0 / Public Domain Canada photos and music for the keepsake demo.

Downloads ~50 Canada landscape/city photos (800px thumbnails), 1 iconic Moraine Lake
target portrait, and 1 CC0 audio track from Wikimedia Commons into `examples/canada/inputs/`.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "keepsake-canada-demo/0.1 (https://github.com/kopib/keepsake)"

TARGET_TITLES = (
    "File:Moraine Lake 17092005.jpg",
    "File:Sunset in the Canadian Rockies (Unsplash).jpg",
)
AUDIO_TITLE = "File:Monplaisir - 01 - Theme Song.ogg"

PHOTO_QUERIES = (
    'Banff OR Jasper OR "Canadian Rockies" incategory:CC-Zero filetype:bitmap',
    '"Quebec City" OR Montreal OR Toronto OR Vancouver incategory:CC-Zero filetype:bitmap',
    'Alberta OR "British Columbia" OR "Lake Louise" OR Niagara incategory:CC-Zero filetype:bitmap',
    "Canada landscape OR lake OR mountain OR autumn incategory:CC-Zero filetype:bitmap",
    'Ontario OR Yukon OR "Nova Scotia" OR Ottawa incategory:CC-Zero filetype:bitmap',
)

EXCLUDED_TITLE_WORDS = (
    "coin",
    "stamp",
    "logo",
    "icon",
    "flag",
    "map",
    "diagram",
    "chart",
    "seal",
    "silhouette",
    "botanique",
    "zinnia",
    "bombus",
    "colorado",
    "denver",
    "coors field",
    "baseball",
    "diamondbacks",
    "portrait",
    "collage",
    "montage",
    "grid",
    "wikipedia",
)

TARGET_PHOTO_COUNT = 50


def _api_get(params: dict[str, str]) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_cc0_or_pd(license_name: str) -> bool:
    low = license_name.strip().lower()
    return low in {"cc0", "public domain", "pd"} or low.startswith("pd-")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def _resolve_file_info(title: str, *, thumb_width: int | None = None) -> tuple[str, str] | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
    }
    if thumb_width is not None:
        params["iiurlwidth"] = str(thumb_width)
    data = _api_get(params)
    for page in data.get("query", {}).get("pages", {}).values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value") or "").strip()
        if not _is_cc0_or_pd(lic):
            continue
        url = (ii.get("thumburl") if thumb_width else ii.get("url")) or ii.get("url")
        if url:
            return url, lic
    return None


def _discover_cc0_photos(limit: int) -> list[tuple[str, str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for query in PHOTO_QUERIES:
        if len(found) >= limit:
            break
        data = _api_get(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": "40",
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "800",
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            low_title = title.lower()
            if (
                title in found
                or title in TARGET_TITLES
                or any(bad in low_title for bad in EXCLUDED_TITLE_WORDS)
            ):
                continue
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            ii = infos[0]
            mime = ii.get("mime", "")
            width = int(ii.get("width", 0) or 0)
            height = int(ii.get("height", 0) or 0)
            lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value") or "").strip()
            url = ii.get("thumburl") or ii.get("url")
            if url and "jpeg" in mime and width >= 600 and height >= 400 and _is_cc0_or_pd(lic):
                found[title] = (url, lic)
                if len(found) >= limit:
                    break
        time.sleep(0.2)

    return [(title, url, lic) for title, (url, lic) in sorted(found.items())]


def main() -> int:
    base_dir = Path(__file__).resolve().parent / "inputs"
    photos_dir = base_dir / "photos"
    audio_dir = base_dir / "audio"
    photos_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    sources_lines = [
        "# Canada Demo Asset Sources (CC0 / Public Domain)",
        "",
        "All demo assets are fetched from Wikimedia Commons under CC0 or Public Domain.",
        "",
    ]

    # 1. Target landscape image
    target_path = base_dir / "target.jpg"
    if not target_path.exists():
        downloaded_target = False
        for title in TARGET_TITLES:
            info = _resolve_file_info(title, thumb_width=1600)
            if info:
                url, lic = info
                print(f"Downloading target image: {title} ({lic})")
                _download(url, target_path)
                sources_lines.append(f"- **Target:** `{title}` ({lic})")
                downloaded_target = True
                break
        if not downloaded_target:
            print("Failed to fetch target image from Wikimedia Commons.", file=sys.stderr)
            return 1
    else:
        print(f"Target image already present: {target_path}")

    # 2. CC0 audio track
    audio_path = audio_dir / "01_theme.ogg"
    if not audio_path.exists():
        info = _resolve_file_info(AUDIO_TITLE)
        if not info:
            print(f"Failed to resolve CC0 audio track: {AUDIO_TITLE}", file=sys.stderr)
            return 1
        url, lic = info
        print(f"Downloading audio track: {AUDIO_TITLE} ({lic})")
        _download(url, audio_path)
        sources_lines.append(f"- **Audio:** `{AUDIO_TITLE}` ({lic})")
    else:
        print(f"Audio track already present: {audio_path}")

    # 3. 50 CC0 Canada photos
    existing = sorted(photos_dir.glob("canada_*.jpg"))
    if len(existing) >= TARGET_PHOTO_COUNT:
        print(f"Already have {len(existing)} demo photos in {photos_dir}")
    else:
        print(f"Searching Wikimedia Commons for {TARGET_PHOTO_COUNT} CC0 Canada photos...")
        candidates = _discover_cc0_photos(TARGET_PHOTO_COUNT)
        sources_lines.append("")
        sources_lines.append(f"## Photos ({len(candidates)})")
        for idx, (title, url, lic) in enumerate(candidates, start=1):
            dest = photos_dir / f"canada_{idx:03d}.jpg"
            if not dest.exists():
                try:
                    _download(url, dest)
                    time.sleep(0.1)
                except Exception as exc:
                    print(f"  Skipped {title}: {exc}", file=sys.stderr)
                    continue
            sources_lines.append(f"- `{dest.name}` — `{title}` ({lic})")
        print(f"Downloaded {len(list(photos_dir.glob('canada_*.jpg')))} photos to {photos_dir}")

    (base_dir / "SOURCES.md").write_text("\n".join(sources_lines) + "\n", encoding="utf-8")
    print(
        f"Done! Now run:\n  uv run keepsake --config {Path(__file__).parent / 'keepsake.toml'} run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
