"""Build report/build/report.html from the chapter Markdown files and the generated figures.

Usage: python -m carryon.viz.report   (``make report``)

The HTML is self-contained: figures are embedded, so the file can be opened or shared on its own.
A figure slot whose image does not exist yet renders as a visible placeholder, not a broken link.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import markdown

REPORT_DIR = Path("report")
CHAPTERS = ("chapter_0.md", "chapter_A.md", "chapter_B.md", "chapter_C.md")
IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)")
MIME = {".png": "image/png", ".svg": "image/svg+xml"}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Carry-On report</title>
<style>
body {{ max-width: 52rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
       font-family: system-ui, sans-serif; color: #222; }}
img {{ max-width: 100%; }}
.missing {{ display: inline-block; padding: 0.4rem 0.7rem; border: 1px dashed #999;
           color: #666; font-size: 0.9rem; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 0.2rem 0.5rem; }}
</style></head><body>
{body}
</body></html>
"""


def build_report(report_dir: Path = REPORT_DIR) -> Path:
    chapters = [(report_dir / name).read_text(encoding="utf-8") for name in CHAPTERS]
    text = "\n\n".join(_embed_images(chapter, report_dir) for chapter in chapters)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    out = report_dir / "build" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(body=body), encoding="utf-8")
    return out


def _embed_images(text: str, base: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt, src = html.escape(match["alt"]), match["src"]
        path = base / src
        if not path.exists():
            return (
                f'<span class="missing">Figure {alt} not generated yet ({html.escape(src)})</span>'
            )
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = MIME.get(path.suffix, "application/octet-stream")
        return f'<img alt="{alt}" src="data:{mime};base64,{data}">'

    return IMAGE.sub(replace, text)


if __name__ == "__main__":
    print(build_report())
