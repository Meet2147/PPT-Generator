from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DIST_DIR = ROOT / "web-dist"


def main() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    shutil.copytree(WEB_DIR, DIST_DIR)

    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    app_name = os.getenv("WEB_APP_NAME", "DeckMint")

    config_js = (
        "window.DECKMINT_CONFIG = "
        f"{{ apiBaseUrl: {api_base_url!r}, appName: {app_name!r} }};\n"
    )
    (DIST_DIR / "config.js").write_text(config_js, encoding="utf-8")


if __name__ == "__main__":
    main()
