"""docs/methodology.md → docs/methodology.pdf (A4, 인쇄용).

  python docs/build_pdf.py
"""
import os
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
@page { size: A4; margin: 16mm 16mm 16mm 16mm; }
:root { --ink:#0b0b0b; --sub:#52514e; --line:#d9d8d3; --accent:#2a78d6; --soft:#f3f6fb; }
body { font-family: "Noto Sans KR", "Malgun Gothic", sans-serif; color: var(--ink); font-size: 9.6pt;
       line-height: 1.6; background: #fff; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 2px; letter-spacing: -0.5px; }
h1 + p { margin-top: 0; }
h2 { font-size: 13pt; margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 2px solid var(--ink);
     break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 6px; color: var(--accent); break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul { padding-left: 18px; margin: 4px 0 8px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.8pt; break-inside: avoid; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: var(--soft); font-weight: 700; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { white-space: nowrap; }
blockquote { margin: 6px 0; padding: 6px 12px; background: var(--soft); border-left: 3px solid var(--accent); }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f2f2ef; padding: 0 3px; border-radius: 3px; }
pre { background: #f2f2ef; padding: 8px 10px; border-radius: 4px; font-size: 8.2pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 8px auto 4px; }
hr { border: none; border-top: 1px solid var(--line); margin: 12px 0; }
strong { font-weight: 700; }
"""


def build(src="methodology.md", out="methodology.pdf"):
    md = open(os.path.join(HERE, src), encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>GameGuard 이상탐지 방법론</title>' \
           f'<style>{CSS}</style></head><body>{body}</body></html>'
    tmp = os.path.join(HERE, "_methodology.html")
    open(tmp, "w", encoding="utf-8").write(html)
    pdf = os.path.join(HERE, out)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file:///" + tmp.replace("\\", "/")], capture_output=True, timeout=120)
    os.remove(tmp)
    return pdf


if __name__ == "__main__":
    print(build())
