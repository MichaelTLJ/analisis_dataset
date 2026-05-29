import json
import shutil

from .config import OUTPUT_DIR, STATIC_DIR, TEMPLATE_DIR


def render_dashboard(payload):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(exist_ok=True)

    shutil.copy2(STATIC_DIR / "css" / "embedding.css", assets_dir / "embedding.css")
    shutil.copy2(STATIC_DIR / "js" / "embedding.js", assets_dir / "embedding.js")

    template = (TEMPLATE_DIR / "embedding_dashboard.html").read_text(encoding="utf-8")
    html = template.replace("__SDSS_PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path

